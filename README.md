# Sparse Encoding Server

## Getting Started

### Run Docker

```
docker run -p 8080:8080 -it ghcr.io/codelibs/sparse-encoding-api:1.0.0
```

### Run Docker with Model Name

```
docker run -p 8080:8080 -e MODEL_NAME=naver/splade_v2_max -it ghcr.io/codelibs/sparse-encoding-api:1.0.0
```

### Use Model Cache

```
docker run -v ./model:/code/model -p 8080:8080 -it ghcr.io/codelibs/sparse-encoding-api:1.0.0
```

### Run Docker with GPU

```
docker run --gpus all -p 8080:8080 -it ghcr.io/codelibs/sparse-encoding-api:1.0.0.cuda11
```

## Request

### Embedding API

```
curl -s -H "Content-Type:application/json" -XPOST localhost:8080/encode -d '
{
  "sentences": [
    "This framework generates embeddings for each input sentence",
    "Sentences are passed as a list of string.",
    "The quick brown fox jumps over the lazy dog."
  ]
}'
```

### Information API

```
curl -s -H "Content-Type:application/json" -XGET localhost:8080/
```

### Ping API

```
curl -s -H "Content-Type:application/json" -XGET localhost:8080/ping
```

## Build

### Build Docker

```
docker build --rm -t ghcr.io/codelibs/sparse-encoding-api:1.0.0 .
```

