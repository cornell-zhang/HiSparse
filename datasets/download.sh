ggID='1VCus77NffWdEfppD5xE6sIIZtx7yNZ6m'
ggURL='https://drive.google.com/uc?export=download'
if [ -f "sparse_datasets.zip" ]; then
    unzip sparse_datasets.zip
else
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${filename}"
    unzip ${filename}
fi
