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
#include <parquet/stream_reader.h>
#include <parquet/stream_writer.h>

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

namespace {

struct ParquetWriterOptions {
  ParquetWriterOptions() {}
};

class ParquetWriter {
 public:
  ParquetWriter(const ParquetWriterOptions& options,
                std::shared_ptr<arrow::io::OutputStream> file);
  void Append(int timestep, int rowid, float v02, float v03);
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
    : writer_(nullptr) {
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

void ParquetWriter::Append(int timestep, int rowid, float v02, float v03) {
  *writer_ << timestep << rowid << v02 << v03 << parquet::EndRow;
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

void Rewrite0(parquet::StreamReader* reader, const std::string& dst) {
  std::shared_ptr<arrow::io::FileOutputStream> file;
  PARQUET_ASSIGN_OR_THROW(file, arrow::io::FileOutputStream::Open(dst))
  ParquetWriter writer(ParquetWriterOptions(), file);
  int n = 0;
  int timestep, rowid;
  float v02, v03;
  while (!reader->eof() && n < 31250000) {
    *reader >> timestep >> rowid >> v02 >> v03 >> parquet::EndRow;
    writer.Append(timestep, rowid, v02, v03);
    n++;
  }
  writer.Finish();
}

void Rewrite(const std::string& src) {
  printf("Rewriting %s to parquet... \n", src.c_str());
  std::shared_ptr<arrow::io::ReadableFile> file;
  PARQUET_ASSIGN_OR_THROW(file, arrow::io::ReadableFile::Open(src));
  parquet::StreamReader reader(parquet::ParquetFileReader::Open(file));
  std::string dst = src;
  int i = 0;
  while (!reader.eof()) {
    dst.resize(src.size());
    dst += ".";
    dst += std::to_string(i++);
    Rewrite0(&reader, dst);
  }
}

void ProcessDir(const char* indir) {
  DIR* const dir = opendir(indir);
  if (!dir) {
    fprintf(stderr, "Fail to open dir %s: %s\n", indir, strerror(errno));
    exit(EXIT_FAILURE);
  }
  std::string tmp = indir;
  size_t base = tmp.size();
  struct dirent* entry = readdir(dir);
  while (entry) {
    if (entry->d_type == DT_REG) {
      const std::string f = entry->d_name;
      if (StringEndWith(f, ".parquet")) {
        tmp.resize(base);
        tmp += '/';
        tmp += f;
        Rewrite(tmp);
      }
    }
    entry = readdir(dir);
  }
  closedir(dir);
  printf("Done!\n");
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s inputdir", argv[0]);
    exit(EXIT_FAILURE);
  }
  ProcessDir(argv[1]);
  return 0;
}
