import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const sourceRoot = path.join(root, 'src', 'content');
const generatedRoot = path.join(root, '.astro', 'obsidian-content');
const publicDir = path.join(root, 'public');

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  const files = [];

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);

    if (entry.isDirectory()) {
      files.push(...walk(fullPath));
    } else if (entry.isFile() && fullPath.endsWith('.md')) {
      files.push(fullPath);
    }
  }

  return files;
}

function listPublicImages() {
  if (!fs.existsSync(publicDir)) return [];

  return fs
    .readdirSync(publicDir)
    .filter((file) => /\.(png|jpe?g|gif|svg|webp|avif)$/i.test(file))
    .sort();
}

function normalizeLink(raw, imageIndexRef, publicImages) {
  if (!raw) return raw;

  const trimmed = raw.trim();

  if (!trimmed || trimmed.startsWith('http://') || trimmed.startsWith('https://') || trimmed.startsWith('/') || trimmed.startsWith('#') || trimmed.startsWith('data:')) {
    return trimmed;
  }

  if (trimmed.startsWith('attachment:')) {
    const filename = trimmed.split(':').at(-1);
    if (filename && publicImages.length) {
      const chosen = publicImages[imageIndexRef.value % publicImages.length];
      imageIndexRef.value += 1;
      return `/${encodeURIComponent(chosen)}`;
    }
    return '/';
  }

  if (/\bimage\.png\b/i.test(trimmed) || /\battachment\b/i.test(trimmed)) {
    if (publicImages.length) {
      const chosen = publicImages[imageIndexRef.value % publicImages.length];
      imageIndexRef.value += 1;
      return `/${encodeURIComponent(chosen)}`;
    }
    return '/';
  }

  return `/${trimmed}`;
}

function copyMarkdownForBuild() {
  if (fs.existsSync(generatedRoot)) {
    fs.rmSync(generatedRoot, { recursive: true, force: true });
  }

  fs.mkdirSync(generatedRoot, { recursive: true });

  const publicImages = listPublicImages();

  function copyDir(sourceDir, targetDir) {
    fs.mkdirSync(targetDir, { recursive: true });

    for (const entry of fs.readdirSync(sourceDir, { withFileTypes: true })) {
      const sourcePath = path.join(sourceDir, entry.name);
      const targetPath = path.join(targetDir, entry.name);

      if (entry.isDirectory()) {
        copyDir(sourcePath, targetPath);
      } else if (entry.isFile() && sourcePath.endsWith('.md')) {
        const original = fs.readFileSync(sourcePath, 'utf8');
        let updated = original;
        const imageIndexRef = { value: 0 };

        updated = updated.replace(/!\[[^\]]*\]\(([^)]+)\)/g, (match, url) => {
          const normalized = normalizeLink(url, imageIndexRef, publicImages);
          return match.replace(url, normalized);
        });

        fs.writeFileSync(targetPath, updated, 'utf8');
      }
    }
  }

  if (fs.existsSync(sourceRoot)) {
    copyDir(sourceRoot, path.join(generatedRoot));
  }

  console.log(`Prepared build-only Obsidian content in ${generatedRoot} using ${publicImages.length} public asset(s)`);
}

copyMarkdownForBuild();
