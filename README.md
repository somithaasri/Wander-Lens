# Itnery Chat

This project consists of a FastAPI backend and a Flutter frontend.

## Backend

The backend is a FastAPI application that generates travel itineraries.

### Requirements

-   Python 3.9
-   pip

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
uvicorn main:app --reload --port 8002
```

```bash
uvicorn Chat:app --reload --port 8001
```

## Frontend

The frontend is a Flutter application that provides a user interface for generating travel itineraries.

### Requirements

-   Flutter SDK

### Installation

```bash
cd dummy_flutter_app
flutter pub get
```

### Running

```bash
cd dummy_flutter_app
flutter run
