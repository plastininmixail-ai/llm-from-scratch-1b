#!/bin/bash
# Скачивает 49 сайтов параллельно

cd "C:/Users/mixai/Desktop/llm-from-scratch"
mkdir -p data/sources/misc_tools
cd data/sources/misc_tools

URLS=(
"unpaywall.org"
"openlibrary.org"
"doaj.org"
"alternativeto.net"
"justwatch.com"
"archive.org"
"gutenberg.org"
"openculture.com"
"wolframalpha.com"
"photopea.com"
"squoosh.app"
"remove.bg"
"cleanup.pictures"
"unscreen.com"
"carbon.now.sh"
"ray.so"
"shots.so"
"smartmockups.com"
"haveibeenpwned.com"
"virustotal.com"
"privnote.com"
"temp-mail.org"
"file.io"
"archive.ph"
"similarsites.com"
"radio.garden"
"everynoise.com"
"tunefind.com"
"musicforprogramming.net"
"mynoise.net"
"coffitivity.com"
"elicit.org"
"consensus.app"
"connectedpapers.com"
"semanticscholar.org"
"scispace.com"
"summarize.tech"
"phind.com"
"regex101.com"
"codebeautify.org"
"jsonformatter.org"
"explainshell.com"
"raindrop.io"
"downdetector.com"
"tineye.com"
"fast.com"
"smallpdf.com"
"ilovepdf.com"
"10minutemail.com"
)

for url in "${URLS[@]}"; do
    slug=$(echo "$url" | sed 's/\./_/g')
    curl -L -o "${slug}.html" "https://${url}" --silent -w "%{http_code} ${url}\n" 2>&1 &
done
wait

echo ""
echo "Downloaded files:"
ls -la *.html 2>/dev/null | wc -l
