===============
Troubleshooting
===============

Common issues and fixes.

ModuleNotFoundError
-------------------

If Python cannot find modules in ``src/``, set the PYTHONPATH:

.. code-block:: bash

    export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

Sphinx build warnings
---------------------

- Missing toctree entries: create the referenced RSTs under ``docs/`` or remove references.
- Title underline errors: ensure the number of underline characters equals the title length.

LaTeX/PDF build errors
----------------------

- If the LaTeX toolchain is missing, install a TeX distribution locally (e.g., TeX Live) and run ``make latexpdf`` in ``docs/_build/latex``.
- For quick PDF export without TeX, use the HTML-to-PDF conversion workflow in the repository (the agent script uses WeasyPrint).

Memory / OOM during training
----------------------------

- Reduce dataset size, increase swap, or run training with fewer `n_trials` and lower `n_estimators`.

Contact
-------

Open an issue on the repository if problems persist.
