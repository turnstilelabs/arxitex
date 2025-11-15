from pathlib import Path

from setuptools import find_packages, setup

here = Path(__file__).parent.resolve()


def read_requirements(path: str = "requirements.txt"):
    req_file = here / path
    if not req_file.exists():
        return []
    lines = req_file.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="arxitex",
    version="0.1.0",
    description="Build searchable dependency graphs from arXiv",
    long_description=(
        (here / "README.md").read_text(encoding="utf-8")
        if (here / "README.md").exists()
        else ""
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/dsleo/arxitex",
    author="dsleo",
    author_email="leo@turnstilelabs.com",
    license="MIT",
    packages=find_packages(exclude=("tests", "pipeline_output", "data")),
    include_package_data=True,
    install_requires=read_requirements(),
    python_requires=">=3.11",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "arxitex=arxitex.workflows.cli:cli_main",
            "arxitex-viz=arxitex.extractor.visualization.cli:main",
        ]
    },
)
