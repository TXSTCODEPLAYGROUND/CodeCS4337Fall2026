Utilities
=========

Helpers for paths, seeding, and run folders. They are also available from the
package itself:

.. code-block:: python

   from TrainingBasicConvnet.utils import (
       get_device,
       resolve_config_path,
       resolve_repo_path,
       set_seed,
       create_run_dir,
       append_run_summary,
   )

``PROJECT_DIR``, ``PROJECT_NAME``, ``REPO_DIR``, and ``CONFIGS_DIR`` in
:mod:`TrainingBasicConvnet.utils.paths` are absolute paths derived from the
location of the package on disk.

.. automodule:: TrainingBasicConvnet.utils.paths
   :members:

.. automodule:: TrainingBasicConvnet.utils.reproducibility
   :members:

.. automodule:: TrainingBasicConvnet.utils.experiment
   :members:
