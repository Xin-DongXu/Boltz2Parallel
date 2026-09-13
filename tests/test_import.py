def test_version():
    from boltz2parallel import __version__

    assert __version__ == "1.0.0"


def test_cli_help():
    from boltz2parallel.cli.main import main

    assert main(["--help"]) == 0
    assert main(["--version"]) == 0
