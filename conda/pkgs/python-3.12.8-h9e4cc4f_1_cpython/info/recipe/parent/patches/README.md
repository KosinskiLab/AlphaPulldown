### How to re-generate patches
```bash
old=v3.9.6
new=v3.10.0
git clone git@github.com:python/cpython && cd cpython
git reset --hard $old
for f in ../recipe/patches/*.patch; do
  git am $f;
done
head=$(git rev-parse HEAD)
git reset --hard $new
git cherry-pick $old...$head  # fix conflicts and make sure the editor doesn't add end of file line ending
git format-patch --no-signature $new
for f in *.patch; do
  python ../recipe/patches/make-mixed-crlf-patch.py $f;
done
```

On windows, the last loop can be done as follows (once the patches
have been copied; executed within the `recipe/patch` folder):
```bash
for /f %f in ('dir /b /S .') do python make-mixed-crlf-patch.py %f
```
