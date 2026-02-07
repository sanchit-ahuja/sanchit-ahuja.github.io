# Copilot Instructions — sanchit-ahuja.github.io

## Project Overview

Static academic personal website for Sanchit Ahuja, built with **Jekyll** and hosted on **GitHub Pages** at `sanchitahuja.com`. Based on [Martin Saveski's template](https://faculty.washington.edu/msaveski/).

## Architecture

- **Jekyll static site** — no Gemfile; relies on GitHub Pages' built-in Jekyll build pipeline.
- **Single-page layout**: [index.html](../index.html) is the only page, with anchor sections (`#bio`, `#publications`, `#resume`). The `#projects` section exists but is commented out.
- **Layouts**: [_layouts/default.html](../_layouts/default.html) (main shell with nav, header, footer) and [_layouts/project.html](../_layouts/project.html) (unused project detail pages).
- **Data-driven content**: all structured content lives in `_data/` YAML files — edit those, not the HTML templates, to update content:
  - `main_info.yaml` — name, title, email, social links, profile pic path
  - `publications.yaml` — papers array with `selected: y/n` flag for tab filtering
  - `experience.yaml` — timeline entries with `category: "work"` or `"education"` (controls left/right placement)
- **CSS framework**: Skeleton (lightweight grid), with custom overrides in `libs/custom/my_css.css`. Grid uses Skeleton's `row`/`columns` classes (e.g., `three columns`, `nine columns`).
- **Icons**: Font Awesome 4.7.0 (`fa fa-*`) and Academicons 1.8.6 (`ai ai-*`).
- **JS**: jQuery 3.1.1 for sticky nav docking, smooth scroll, tab switching (skeleton-tabs), and popover handling. Custom logic in `libs/custom/my_js.js`.

## Key Conventions

### Adding a new publication
Add an entry to `_data/publications.yaml` following this exact structure:
```yaml
- title: "Paper Title"
  authors: "<b>Sanchit Ahuja</b>, Co-Author Name"
  venue: "Conference Name (YEAR)"
  paper_pdf: "https://link-to-paper"
  code: "https://github.com/repo"       # optional
  slides: "/assets/publications/..."     # optional
  selected: y                            # y = appears in "Selected" tab
```
Bold the site owner's name with `<b>` tags. Use `<sup>‡</sup>` for equal contribution markers.

### Adding a new experience
Add to `_data/experience.yaml`:
```yaml
- place: "Organization"
  time: "YYYY-YYYY"
  title: "Role"
  subtitle: "Focus areas"
  category: "work"    # "work" = left side, "education" = right side of timeline
```

### Asset organization
Publication assets go under `assets/publications/<year>_<short_name>/`. Profile pictures in `assets/profile-pics/`. CV files in `assets/cv/`.

### Styling patterns
- Container max-width is `800px` (set in `my_css.css`, not Skeleton defaults).
- Navbar is hidden on mobile (`display: none`) and shown as a sticky docked bar above `750px` viewport width.
- Font: Inter (Google Fonts), replacing the original Raleway.
- Accent color: `#33C3F0` (Skeleton's default blue, used for active nav links and hover states).

## Development Workflow

- **Local preview**: `jekyll serve` (requires Ruby + Jekyll gem installed locally).
- **Deploy**: push to `main` branch — GitHub Pages auto-builds. The `__deploy.sh` script is a leftover from the original template (uploads via `scp`) and is **not used**.
- **No `baseurl`** is configured in `_config.yml`; the site serves from the root.
- **Custom domain**: `sanchitahuja.com` (configured via `CNAME` file).

## Things to Watch Out For

- The Google Fonts `<link>` tag in `default.html` has a malformed attribute (mixed quotes in `href`). Avoid introducing similar issues.
- Google Analytics uses the legacy `analytics.js` (not gtag.js) and reads the tracking ID from `main_info.yaml` key `google_analystics_tracking_id` (note the typo — keep it consistent).
- The `_data/template_users.yaml` and `_data/projects.yaml` are template remnants — the projects section is commented out in `index.html`.
- `_collections: projects` is defined in `_config.yml` but the `_projects/` directory doesn't exist — this is unused.
