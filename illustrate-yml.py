import yaml
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_from_file(
    yml_path: str,
    env_name: str,
):
    with open(yml_path, mode="rt", encoding="utf-8") as file:
        yml_config = yaml.safe_load(file)

    accounts = yml_config.get("accounts", [])

    for account_info in accounts:
        logger.info("building '%s'", account_info.get("name"))

    env = yml_config.get("env", {})
    env_config = env.get(env_name)
    logger.info("ENV Config: %s", env_config)

    logger.info("NETWORK: %s", yml_config.get("network"))


"""

-- you can do something like the following in your docker file
--
RUN echo "network: $NETWORK" > /network.yml

ENTRYPOINT ["python", "your_python_file.py", "-c", "network.yml"]


"""

if __name__ == "__main__":
    load_from_file(
        yml_path = "xrpl.yml",
        env_name = "mainnet",
    )
