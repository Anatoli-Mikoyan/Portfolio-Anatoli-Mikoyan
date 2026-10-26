# Source du CV

`cv.html` est la source du CV publié à la racine (`Cv.pdf`).

Pour régénérer le PDF après modification :

```bash
# depuis la racine du dépôt
npx http-server -p 8899 -s &
npx playwright@1.56 screenshot --help >/dev/null   # s'assure que Chromium est présent
node -e "
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('http://127.0.0.1:8899/cv/cv.html', { waitUntil: 'networkidle' });
  await p.pdf({ path: 'Cv.pdf', format: 'A4', printBackground: true,
                margin: { top: 0, right: 0, bottom: 0, left: 0 } });
  await b.close();
})();"
```

Le contenu doit tenir sur **une seule page A4** : le bloc `.page` ne doit pas
dépasser 1123 px de haut à l'écran. Au-delà, le PDF passe à deux pages.
