Dependencies:

This project requires python 3.10 and above

`pip install fastapi pandas numpy torch openpyxl uvicorn`

===

AFTER installing the dependencies:

In the current directory, run:

`uvicorn myapi:app --reload`

The app should be served by default on localhost:8000

===

the docs are available at localhost:8000/docs

===

Usage:

- 3 endpoints, described in app.py
- data in POST routes should strictly adhere to the structure documented by the examples. (Strict validation enabled with error msg etc)