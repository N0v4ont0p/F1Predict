"""Dev panel: a localhost job runner for the checks and the M6 checklist, driven from the web app.

``f1p dev serve`` starts it; the web app's ``/dev`` page lists the tasks, starts them, and
streams their output live. Only the tasks in :mod:`f1p.devpanel.tasks` can run — never an
arbitrary command — and the server listens on 127.0.0.1 only.
"""
