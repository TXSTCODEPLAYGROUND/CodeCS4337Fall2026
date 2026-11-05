"""This project's paths, configs, and run lookup, from the ``colabutilscs4337`` library."""

from pathlib import Path

from colabutilscs4337.project import Project

PROJECT = Project(Path(__file__).resolve().parents[1])
"""The :class:`~colabutilscs4337.project.Project` of this folder.

It knows the repo root (``.env``, ``data/``, ``runs/``), the ``configs/`` folder, and
where runs are saved: ``<OUTPUT_DIR>/LitWBTransferLearningResnet/<config>/<timestamp>/``.
The path, config, and run-lookup code is shared by every project, so it lives in
`colabutilscs4337 <https://txstcodeplayground.github.io/ColabUtilsCS4337/guide/course_projects.html>`__.
"""
