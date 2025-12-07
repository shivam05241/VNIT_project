================
Deployment Guide
================

This guide walks through containerizing and running the app with Docker.

Docker (quick)
---------------

Build the image:

.. code-block:: bash

    docker build -t vnit-housing:latest .

Run the container (exposes Streamlit app):

.. code-block:: bash

    docker run -p 8501:8501 vnit-housing:latest

Docker Compose
--------------

A ``docker-compose.yml`` is included for convenience. Start the stack:

.. code-block:: bash

    docker-compose up --build

Notes
-----

- Ensure model artifact ``models/xgboost_model.joblib`` and any reports are accessible to the container (they're in the repo).
- For production, push the image to your registry and deploy to your cloud provider.
