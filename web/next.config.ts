import type { NextConfig } from "next";

const config: NextConfig = {
  output: "export",
  trailingSlash: true,
  reactCompiler: true,
  images: { unoptimized: true },
};

export default config;
