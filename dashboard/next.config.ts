import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  // TSAI Jarvis Dashboard runs on port 8000
  // This avoids conflicts with other TSAI services
  // See docs/product-development/APPLICATION-PORTS-MAPPING.md
};

export default nextConfig;
