/*
 * Copyright (c) 2024 Triad National Security, LLC, as operator of Los Alamos
 * National Laboratory with the U.S. Department of Energy/National Nuclear
 * Security Administration. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of TRIAD, Los Alamos National Laboratory, LANL, the
 *    U.S. Government, nor the names of its contributors may be used to endorse
 *    or promote products derived from this software without specific prior
 *    written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/stream_writer.h>

#include <vtkDataArraySelection.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPointData.h>
#include <vtkXMLImageDataReader.h>

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <unordered_map>

namespace {

class Iterator {
 public:
  explicit Iterator(vtkImageData* image);
  ~Iterator() {}

  void SeekToFirst() { i_ = 0; }
  bool Valid() const { return i_ >= 0 && i_ < n_; }
  void Next() { i_++; }

  float v02() const { return v02_[i_]; }
  float v03() const { return v03_[i_]; }

 private:
  int n_;  // Total number of elements
  float* v02_;
  float* v03_;
  int i_;
};

Iterator::Iterator(vtkImageData* image) {
  vtkPointData* const pointData = image->GetPointData();
  n_ = image->GetNumberOfPoints();
#define GET_POINTER(pointData, attributename)                             \
  vtkFloatArray::FastDownCast(pointData->GetAbstractArray(attributename)) \
      ->GetPointer(0)
  v02_ = GET_POINTER(pointData, "v02");
  v03_ = GET_POINTER(pointData, "v03");
#undef GET_POINTER
  i_ = 0;
}

struct ParquetWriterOptions {
  ParquetWriterOptions(int rowid) : rowid(rowid) {}
  int32_t rowid;
};

class ParquetWriter {
 public:
  ParquetWriter(const ParquetWriterOptions& options,
                std::shared_ptr<arrow::io::OutputStream> file);
  void Append(int timestep, float v02, float v03);
  void Finish();
  ~ParquetWriter();

 private:
  // No copying allowed
  ParquetWriter(const ParquetWriter&);
  void operator=(const ParquetWriter& other);
  parquet::StreamWriter* writer_;
  int32_t rowid_;
};

namespace {
std::shared_ptr<parquet::schema::GroupNode> GetSchema() {
  parquet::schema::NodeVector fields;
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "timestep", parquet::Repetition::REQUIRED, parquet::Type::INT32,
      parquet::ConvertedType::INT_32));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "rowid", parquet::Repetition::REQUIRED, parquet::Type::INT32,
      parquet::ConvertedType::INT_32));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "v02", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "v03", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  return std::static_pointer_cast<parquet::schema::GroupNode>(
      parquet::schema::GroupNode::Make("schema", parquet::Repetition::REQUIRED,
                                       fields));
}
}  // namespace

ParquetWriter::ParquetWriter(const ParquetWriterOptions& options,
                             std::shared_ptr<arrow::io::OutputStream> file)
    : writer_(nullptr), rowid_(options.rowid) {
  parquet::WriterProperties::Builder builder;
  builder.compression("timestep", parquet::Compression::SNAPPY);
  builder.compression("rowid", parquet::Compression::SNAPPY);
  builder.compression(parquet::Compression::UNCOMPRESSED);
  builder.encoding("timestep", parquet::Encoding::DELTA_BINARY_PACKED);
  builder.encoding("rowid", parquet::Encoding::DELTA_BINARY_PACKED);
  builder.encoding(parquet::Encoding::PLAIN);
  builder.disable_dictionary();
  writer_ = new parquet::StreamWriter(parquet::ParquetFileWriter::Open(
      std::move(file), GetSchema(), builder.build()));
}

void ParquetWriter::Append(int timestep, float v02, float v03) {
  *writer_ << timestep << rowid_++ << roundf(v02 * 1000000) / 1000000
           << roundf(v03 * 1000000) / 1000000 << parquet::EndRow;
}

void ParquetWriter::Finish() {
  delete writer_;
  writer_ = nullptr;
}

ParquetWriter::~ParquetWriter() { delete writer_; }

}  // namespace

inline bool StringEndWith(const std::string& str, const char* suffix) {
  std::size_t lenstr = str.length();
  std::size_t lensuffix = std::strlen(suffix);
  if (lensuffix > lenstr) {
    return false;
  }
  return std::strncmp(&str[0] + lenstr - lensuffix, suffix, lensuffix) == 0;
}

int Rewrite0(int timestep, int rowid, Iterator* it, const std::string& from,
             const std::string& to) {
  std::shared_ptr<arrow::io::FileOutputStream> file;
  PARQUET_ASSIGN_OR_THROW(file, arrow::io::FileOutputStream::Open(to))
  ParquetWriter writer(ParquetWriterOptions(rowid), file);
  int n = 0;
  while (it->Valid() && n < 100 * 500 * 500) {
    writer.Append(timestep, it->v02(), it->v03());
    n++;
    it->Next();
  }
  writer.Finish();
  return n;
}

void Rewrite(int timestep, const std::string& from, const std::string& to) {
  printf("Rewriting %s to parquet... \n", from.c_str());
  vtkNew<vtkXMLImageDataReader> reader;
  reader->SetFileName(from.c_str());
  reader->UpdateInformation();
  vtkDataArraySelection* das = reader->GetPointDataArraySelection();
  das->DisableAllArrays();
  das->EnableArray("v02");
  das->EnableArray("v03");
  reader->Update();
  vtkImageData* image = reader->GetOutput();
  Iterator it(image);
  it.SeekToFirst();
  std::string myto = to;
  int i = 0;
  int rowid = 0;
  while (it.Valid()) {
    myto.resize(to.size());
    myto += ".";
    myto += std::to_string(i);
    i++;
    rowid += Rewrite0(timestep, rowid, &it, from, myto);
  }
}

void ProcessDir(const char* indir, const char* outdir) {
  DIR* const dir = opendir(indir);
  if (!dir) {
    fprintf(stderr, "Fail to open dir %s: %s\n", indir, strerror(errno));
    exit(EXIT_FAILURE);
  }
  std::string tmp1 = indir;
  size_t base1 = tmp1.size();
  std::string tmp2 = outdir;
  size_t base2 = tmp2.size();
  struct dirent* entry = readdir(dir);
  while (entry) {
    if (entry->d_type == DT_REG) {
      const std::string f = entry->d_name;
      if (StringEndWith(f, ".vti")) {
        tmp1.resize(base1);
        tmp1 += '/';
        tmp1 += f;
        tmp2.resize(base2);
        tmp2 += '/';
        tmp2 += f.substr(0, f.size() - 4);
        tmp2 += ".parquet";
        int t = atoi(f.substr(f.size() - 4 - 5, 5).c_str());
        Rewrite(t, tmp1, tmp2);
      }
    }
    entry = readdir(dir);
  }
  closedir(dir);
  printf("Done!\n");
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s inputdir <outputdir>", argv[0]);
    exit(EXIT_FAILURE);
  }
  ProcessDir(argv[1], argc > 2 ? argv[2] : ".");
  return 0;
}
