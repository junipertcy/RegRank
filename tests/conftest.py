# In tests/conftest.py
import subprocess
import time
from pathlib import Path

import pymongo
import pytest
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError


@pytest.fixture(scope="session", autouse=True)
def cleanup_docker_container_before_session():
    """
    A session-scoped fixture to forcefully remove the 'mongodb' container
    before any tests are run.
    """
    container_name = "mongodb"
    # This command forcefully removes the container (if it exists).
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)  # noqa: S603, S607
    return None  # Explicitly return None for clarity


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    """Returns the absolute path to the docker-compose.yml file."""
    root_path = getattr(pytestconfig, "root_path", pytestconfig.rootdir)
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

    client = None
    try:
        # Give the service up to 30 seconds to become responsive.
        for _ in range(15):
            try:
                # Increase the timeout for server selection.
                client = pymongo.MongoClient(auth_url, serverSelectionTimeoutMS=5000)
                # The ismaster command is lightweight and effective for checking readiness.
                client.admin.command("ismaster")
                print("\n--> MongoDB is responsive! Handing over to tests.")
                yield client
                # Exit the loop and fixture setup once connection is successful.
                return
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                print(f". (Retrying due to: {e})", end="", flush=True)
                # Give the container a bit more time to initialize.
                time.sleep(2)

        # If the loop finishes without returning, the service failed to start.
        pytest.fail("MongoDB service did not become responsive within 30 seconds.")

    finally:
        # This teardown code will always run after the test session.
        if client:
            print("\n--- CLOSING mongo_service fixture ---")
            client.close()
