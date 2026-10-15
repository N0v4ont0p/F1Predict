import "@fontsource/titillium-web/400.css";
import "@fontsource/titillium-web/600.css";
import "@fontsource/titillium-web/700.css";
import "@fontsource-variable/inter-tight";
import "@fontsource-variable/jetbrains-mono";
import "./globals.css";
import type { Metadata, Viewport } from "next";
import Link from "next/link";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: { default: "f1predict — probabilistic Formula 1 forecasts", template: "%s · f1predict" },
  description:
    "Calibrated, fully probabilistic Formula 1 race forecasts: every car's finishing-position distribution, before the weekend and after qualifying.",
};

export const viewport: Viewport = { themeColor: "#0b0c10", colorScheme: "dark" };

const NAV = [
  ["/", "Hub"],
  ["/season/", "Season"],
  ["/championship/", "Title"],
  ["/ratings/", "Ratings"],
  ["/lab/", "What-if"],
  ["/model/", "Model Lab"],
] as const;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="relative">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4"
        >
          Skip to content
        </a>
        <header className="sticky top-0 z-30 border-b border-white/5 bg-[oklch(0.12_0.006_265/0.7)] backdrop-blur-xl">
          <nav className="mx-auto flex max-w-7xl items-center justify-between gap-3 px-4 py-3 sm:px-6">
            <Link href="/" className="group flex shrink-0 items-center gap-2.5">
              <span className="relative grid h-7 w-7 place-items-center overflow-hidden rounded-lg bg-[var(--color-carbon-800)]">
                <span className="absolute inset-0 animate-sweep bg-gradient-to-r from-transparent via-[var(--color-signal)]/40 to-transparent" />
                <span className="relative font-mono text-[0.7rem] font-bold text-[var(--color-signal)]">
                  f1
                </span>
              </span>
              <span className="font-display text-lg font-bold tracking-tight">
                f1<span className="text-[var(--color-signal)]">predict</span>
              </span>
            </Link>
            <ul className="-mr-2 flex min-w-0 items-center gap-0.5 overflow-x-auto text-xs [scrollbar-width:none] sm:mr-0 sm:gap-1 sm:text-sm">
              {NAV.map(([href, label]) => (
                <li key={href}>
                  <Link
                    href={href}
                    className="whitespace-nowrap rounded-full px-2 py-1.5 text-[var(--color-carbon-300)] transition hover:bg-white/5 hover:text-white sm:px-3"
                  >
                    {label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>
        </header>
        <main id="main" className="relative z-10 mx-auto max-w-7xl px-4 pb-24 pt-8 sm:px-6">
          {children}
        </main>
        <footer className="relative z-10 border-t border-white/5">
          <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-8 text-xs text-[var(--color-carbon-500)] sm:px-6">
            <p>
              Unofficial and not associated with Formula 1, the FIA or any team. Forecasts are
              probabilities, not advice.
            </p>
            <p>
              Data from F1DB (github.com/f1db/f1db, CC BY 4.0) · Weather data by Open-Meteo.com (CC
              BY 4.0).
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
