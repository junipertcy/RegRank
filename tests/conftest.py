# In tests/conftest.py
from pathlib import Path
import pytest
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import time


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Returns the absolute path to the docker-compose.yml file."""
    root_path = getattr(pytestconfig, "root_path", getattr(pytestconfig, "rootdir"))
    return str(Path(str(root_path)) / "services" / "docker-compose.yml")

@pytest.fixture(scope="session")
def docker_setup():
    """Defines the command to start the Docker services."""
    return ["down -v --remove-orphans", "up --build -d"]

@pytest.fixture(scope="session")
def docker_cleanup():
    """Defines the command to stop the Docker services."""
    return "down -v --remove-orphans"

@pytest.fixture(scope="session")
def mongo_service(docker_services):
    """
    Waits for MongoDB to be responsive, provides a client connection,
    and guarantees the client is closed after the test session.
    """
    print("\n--- ENTERING mongo_service fixture ---")
    host = "localhost"
    port = docker_services.port_for("database", 27017)
    user = "tzuchi"
    password = "FI4opd12cjazqPL"
    auth_url = f"mongodb://{user}:{password}@{host}:{port}/"

    # --- Setup Phase ---
    client = None
    for i in range(15):
        try:
            client = pymongo.MongoClient(auth_url, serverSelectionTimeoutMS=1000)
            client.admin.command('ping')
            print("\n--> MongoDB is responsive! Handing over to tests.")
            # Yield the authenticated client to the tests
            yield client
            # The test session runs here
            break # Exit the loop once connection is successful
        except (ConnectionFailure, ServerSelectionTimeoutError):
            print(".", end="", flush=True)
            time.sleep(1)
    else: # This 'else' belongs to the 'for' loop
        pytest.fail("MongoDB service did not become responsive within 15 seconds.")

    # --- Teardown Phase ---
    # This code runs after the entire test session is complete
    if client:
        print("\n--- CLOSING mongo_service fixture ---")
        client.close()
