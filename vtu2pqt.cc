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
#include <parquet/stream_writer.h>

#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkNew.h>
#include <vtkUnstructuredGrid.h>
#include <vtkXMLUnstructuredGridReader.h>

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace xrage {

class Iterator {
 public:
  explicit Iterator(vtkUnstructuredGrid* grid);
  ~Iterator() {}

  void SeekToFirst() { i_ = 0; }
  bool Valid() const { return i_ >= 0 && i_ < n_; }
  void Next() { i_++; }

  float rho() const { return rho_[i_]; }
  float prs() const { return prs_[i_]; }
  float tev() const { return tev_[i_]; }
  float xdt() const { return xdt_[i_]; }
  float ydt() const { return ydt_[i_]; }
  float zdt() const { return zdt_[i_]; }
  float snd() const { return snd_[i_]; }
  float grd() const { return grd_[i_]; }
  float mat() const { return mat_[i_]; }
  float v02() const { return v02_[i_]; }
  float v03() const { return v03_[i_]; }

 private:
  int n_;  // Total number of elements
  float* rho_;
  float* prs_;
  float* tev_;
  float* xdt_;
  float* ydt_;
  float* zdt_;
  float* snd_;
  float* grd_;
  float* mat_;
  float* v02_;
  float* v03_;
  int i_;
};

Iterator::Iterator(vtkUnstructuredGrid* grid) {
  vtkCellData* const celldata = grid->GetCellData();
  n_ = grid->GetNumberOfCells();
#define GET_POINTER(celldata, fieldname)                             \
  vtkFloatArray::FastDownCast(celldata->GetAbstractArray(fieldname)) \
      ->GetPointer(0)
  rho_ = GET_POINTER(celldata, "rho");
  prs_ = GET_POINTER(celldata, "prs");
  tev_ = GET_POINTER(celldata, "tev");
  xdt_ = GET_POINTER(celldata, "xdt");
  ydt_ = GET_POINTER(celldata, "ydt");
  zdt_ = GET_POINTER(celldata, "zdt");
  snd_ = GET_POINTER(celldata, "snd");
  grd_ = GET_POINTER(celldata, "grd");
  mat_ = GET_POINTER(celldata, "mat");
  v02_ = GET_POINTER(celldata, "v02");
  v03_ = GET_POINTER(celldata, "v03");
#undef GET_POINTER
  i_ = 0;
}

struct ParquetWriterOptions {
  ParquetWriterOptions() {}
};

class ParquetWriter {
 public:
  ParquetWriter(const ParquetWriterOptions& options,
                std::shared_ptr<arrow::io::OutputStream> file);
  void Append(Iterator* it);
  void Finish();
  ~ParquetWriter();

 private:
  // No copying allowed
  ParquetWriter(const ParquetWriter&);
  void operator=(const ParquetWriter& other);
  parquet::StreamWriter* writer_;
};

namespace {
std::shared_ptr<parquet::schema::GroupNode> GetSchema() {
  parquet::schema::NodeVector fields;
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "rho", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "prs", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "tev", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "xdt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "ydt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "zdt", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "snd", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "grd", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
  fields.push_back(parquet::schema::PrimitiveNode::Make(
      "mat", parquet::Repetition::REQUIRED, parquet::Type::FLOAT,
      parquet::ConvertedType::NONE));
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
                             std::shared_ptr<arrow::io::OutputStream> file) {
  parquet::WriterProperties::Builder builder;
  builder.compression(parquet::Compression::ZSTD);
  // builder.disable_dictionary();
  writer_ = new parquet::StreamWriter(parquet::ParquetFileWriter::Open(
      std::move(file), GetSchema(), builder.build()));
}

void ParquetWriter::Append(Iterator* it) {
  *writer_ << it->rho() << it->prs() << it->tev() << it->xdt() << it->ydt()
           << it->zdt() << it->snd() << it->grd() << it->mat() << it->v02()
           << it->v03() << parquet::EndRow;
}

void ParquetWriter::Finish() {
  delete writer_;
  writer_ = NULL;
}

ParquetWriter::~ParquetWriter() { delete writer_; }

}  // namespace xrage

inline bool StringEndWith(const std::string& str, const char* suffix) {
  std::size_t lenstr = str.length();
  std::size_t lensuffix = std::strlen(suffix);
  if (lensuffix > lenstr) {
    return false;
  }
  return std::strncmp(&str[0] + lenstr - lensuffix, suffix, lensuffix) == 0;
}

void Rewrite(const std::string& from, const std::string& to) {
  printf("Rewriting %s to parquet... \n", from.c_str());
  vtkNew<vtkXMLUnstructuredGridReader> reader;
  reader->SetFileName(from.c_str());
  reader->Update();
  vtkUnstructuredGrid* grid = reader->GetOutput();
  std::shared_ptr<arrow::io::FileOutputStream> file;
  PARQUET_ASSIGN_OR_THROW(file, arrow::io::FileOutputStream::Open(to))
  xrage::ParquetWriter writer(xrage::ParquetWriterOptions(), file);
  xrage::Iterator it(grid);
  it.SeekToFirst();
  while (it.Valid()) {
    writer.Append(&it);
    it.Next();
  }
  writer.Finish();
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
    if (entry->d_type == DT_REG || entry->d_type == DT_LNK) {
      const std::string f = entry->d_name;
      if (StringEndWith(f, ".vtu")) {
        tmp1.resize(base1);
        tmp1 += '/';
        tmp1 += f;
        tmp2.resize(base2);
        tmp2 += '/';
        tmp2 += f.substr(0, f.size() - 4);
        tmp2 += ".parquet";
        Rewrite(tmp1, tmp2);
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
