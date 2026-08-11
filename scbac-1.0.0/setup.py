from pathlib import Path
import os
from setuptools import setup, find_packages
from setuptools.command.install import install

ROOT = Path(__file__).resolve().parent
requirements = [line.strip() for line in (ROOT / "requirements.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]


class InstallWithModels(install):
    """Best-effort Zenodo download for classic/source installs.

    Modern pip normally builds and installs a wheel, where setuptools' install
    command is not guaranteed to run. Therefore scBAC also guarantees automatic
    model installation on first pretrained-model use.
    """
    def run(self):
        super().run()
        if os.environ.get("SCBAC_SKIP_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
            return
        try:
            from scbac.install_models import install_pretrained_models
            install_pretrained_models(quiet=False)
        except Exception as exc:
            print("WARNING: scBAC pretrained model download during installation did not complete: {}".format(exc))
            print("The models will be downloaded automatically on first pretrained prediction, or run `scbac models install`.")


setup(
    name="scbac",
    version="1.0.0",
    description="Single-cell Brain Age Clocks: train, predict, and analyze RAA/AASO",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Jianfeng Luo, Ganqiang Liu, Yamei Tang",
    author_email="luojf35@mail.sysu.edu.cn",
    url="https://github.com/sixguns1984/scBACs",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"scbac": ["pretrained_raa/*", "pretrained_raa/*/*"]},
    python_requires=">=3.9,<3.12",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "scbac=scbac.cli:main",
            "scbac-install-models=scbac.install_models:main",
        ]
    },
    cmdclass={"install": InstallWithModels},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
