from streamlit.web import cli


def main():
    cli.main_run(["src/clients/streamlit/app.py"])


if __name__ == "__main__":
    main()
