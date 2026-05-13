# Guia de Despliegue: GitHub Pages + Shinylive

Esta guia explica paso a paso como desplegar el proyecto en GitHub Pages utilizando Shinylive para que el dashboard sea accesible desde cualquier navegador sin necesidad de instalar Python.

---

## Opcion 1: GitHub Pages con Shinylive (Recomendado)

Shinylive permite exportar aplicaciones Shiny como archivos estaticos HTML/JS que se ejecutan completamente en el navegador (WebAssembly/Pyodide).

### Paso 1: Preparar el proyecto

```bash
# Instalar shinylive
pip install shinylive
```

### Paso 2: Exportar la aplicacion

```bash
# Desde la raiz del proyecto
shinylive export dashboard/ docs/
```

Esto creara una carpeta `docs/` con los archivos estaticos.

### Paso 3: Configurar GitHub Pages

1. Crear un repositorio en GitHub (ej. `azucar-predictivo`)
2. Subir el proyecto al repositorio:
   ```bash
   git init
   git add .
   git commit -m "Primer commit: proyecto completo"
   git branch -M main
   git remote add origin https://github.com/tu-usuario/azucar-predictivo.git
   git push -u origin main
   ```
3. Ir a **Settings > Pages** en GitHub
4. En "Source" seleccionar **Deploy from a branch**
5. Branch: `main`, folder: `/docs`
6. Guardar

### Paso 4: Acceder

La aplicacion estara disponible en:
```
https://tu-usuario.github.io/azucar-predictivo/
```

---

## Opcion 2: GitHub Actions (Automatizado)

Crear `.github/workflows/deploy.yml`:

```yaml
name: Deploy Shiny Dashboard

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install shinylive
      - run: shinylive export dashboard/ docs/
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/
      - uses: actions/deploy-pages@v4
```

Luego ir a **Settings > Pages** y seleccionar "GitHub Actions" como source.

---

## Opcion 3: Render (Alternativa gratuita)

1. Crear cuenta en https://render.com
2. Conectar repositorio de GitHub
3. Crear nuevo "Web Service"
4. Configuracion:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `shiny run dashboard/app.py --host 0.0.0.0 --port 10000`

---

## Opcion 4: Ejecucion local

```bash
# Con entorno virtual activado
python -m shiny run dashboard/app.py --reload
# Abrir http://localhost:8000
```

---

## Estructura final para GitHub Pages

```
tu-repositorio/
├── docs/                      # Generado por shinylive export
│   ├── index.html
│   ├── shinylive.js
│   └── ... (assets)
├── dashboard/
│   └── app.py
├── scripts/
├── models/
├── ... (resto del proyecto)
```

**Nota:** La carpeta `docs/` es la que GitHub Pages utiliza para servir el sitio.

---

## Verificacion del despliegue

1. Despues del deploy, esperar 1-2 minutos
2. Ir a `https://tu-usuario.github.io/azucar-predictivo/`
3. Verificar que el dashboard carga correctamente
4. Probar los filtros y graficas interactivas
