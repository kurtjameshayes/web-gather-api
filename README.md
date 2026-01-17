# web-gather-api

Flask API for gathering, ingesting, indexing, and searching web documents.

## Setup

1. Create a `.env` file in the repository root:

```
TAVILY_API_KEY=your_tavily_key
MONGODB_URI=mongodb://localhost:27017
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the server:

```
python app.py
```

## Endpoints

- `POST /gather`
- `POST /ingest`
- `GET /documents`
- `GET /collections`
- `POST /index`
- `POST /search`
- `GET /embedding-models`
- `POST /embedding-models`