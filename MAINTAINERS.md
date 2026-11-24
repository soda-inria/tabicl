# Maintainer notes

## Release process

TabICL is published to [PyPI](https://pypi.org/project/tabicl/) by
[`.github/workflows/publish-to-pypi.yml`](.github/workflows/publish-to-pypi.yml).
Publishing uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/):
GitHub Actions mints a short-lived OpenID Connect token, and PyPI exchanges it
for upload credentials. The PyPI and GitHub Release jobs follow joblib/loky
and joblib/threadpoolctl.

The package version is `__version__` in `src/tabicl/__about__.py`. Hatch reads
that file. Between releases the version is the next release number plus a
`.dev0` suffix.

### Making a release

1. Add a `CHANGES.md` section whose heading is the release version on its own
   line, followed by a line of `=` characters. The workflow copies that
   section into the GitHub Release and shortens it if it exceeds GitHub's
   limit. A missing section does not stop the release; the release notes are
   then the full `CHANGES.md` file, without shortening.
2. Set `__version__` in `src/tabicl/__about__.py` to that same version, for
   example `2.3.0`.
3. Commit those changes on `main`.
4. Tag that commit and push the tag. The tag must be `v` plus the version in
   `src/tabicl/__about__.py`, as with `v2.2.0`. The workflow stops before
   uploading when the tag and the version differ. Pushing the tag starts the
   publish workflow, which also creates the GitHub Release.

   ```bash
   git tag v2.3.0
   git push origin v2.3.0
   ```

5. Approve the `pypi` environment when GitHub requests it.
6. The workflow then builds the sdist and wheel, uploads them to PyPI, signs
   them with Sigstore, creates the GitHub Release using the `CHANGES.md`
   section as the release notes, and uploads the distributions and signatures.
   A later run of the same tag keeps an existing release and any assets
   already uploaded.
7. On `main`, set `__version__` to the following release with a `.dev0`
   suffix, for example `2.4.0.dev0` or `2.3.1.dev0`, and commit that change.
