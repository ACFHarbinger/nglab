# ADR-0009: Use Tailwind CSS

## Status
Accepted

## Context
NGLab requires a modern, responsive, and consistent UI design system. Traditional CSS or SASS often leads to style conflicts and "append-only" stylesheets that grow indefinitely. We need a styling solution that enables rapid development, integrates well with React components, and keeps the bundle size small.

## Decision
We will use **Tailwind CSS** (v4) as our primary styling framework.
- **Utility-First**: We will use predefined utility classes (`flex`, `p-4`, `text-red-500`) directly in JSX.
- **Dark Mode**: We will support dark mode natively using Tailwind's `dark:` modifier, aligning with the financial terminal aesthetic.
- **Configuration**: Theme customization (colors, fonts) will be managed centrally in the Tailwind config.

## Consequences
- **Easier**:
    - **Speed**: No context switching between `.tsx` and `.css` files.
    - **Consistency**: "Magic numbers" are replaced by standard spacing/sizing tokens.
    - **Maintenance**: Deleting a component automatically removes its styles (no dead CSS).
- **Difficult**:
    - **Readability**: JSX can become cluttered with long class strings (mitigated by component extraction or `cn()` utilities).
    - **Learning Curve**: Developers must learn the Tailwind utility names.

## Alternatives Considered
- **CSS-in-JS (Styled Components/Emotion)**: Adds runtime overhead and increases bundle size. Tailwind is zero-runtime.
- **SASS/Modules**: Solves scope issues but doesn't enforce a design system or reduce file switching.
- **Component Libraries (MUI/AntD)**: Too opinionated and heavy. We prefer building our own lightweight components or using headless primitives (Radix) styled with Tailwind.
