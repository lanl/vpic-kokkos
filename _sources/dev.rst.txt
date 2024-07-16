Developers
==========

Updating GitHub Pages Documentation
***********************************

The online documentation comes from the branch gh-pages and is not automatically updated when devel or any other branch is updated.  There are ways to automate this, but they seemed fragile, so for now some manual intervention is required every time the documentation is updated.

First, update the documentation in the devel branch and compile the html.  Next, in a separate repo, checkout the gh-pages branch.  Copy all the compiled files (likely in `vpic-kokkos/docs/build/html/` directly into the root of the gh-pages repo.  Don't get cute and copy "only what's updated" because you're liable to mess up links and such.

Open it up with a browser and make sure everything looks fine.  Add all the files ``git add -A``, but watch out for any lingering submodules (Kokkos).  Make a commit and finish it off with a ``git push origin gh-pages``.  It should take immediate effect.  If you don't see the updates or only some of them, your browser might be cacheing some of the pages.
