from src.config import Config


def main():
    try:
        Config.check_directory_presence()
    except Exception as e:
        raise Exception(f"An error occurred while checking directories: {e}")

    print("Hello from nlp-classification!")


if __name__ == "__main__":
    main()
