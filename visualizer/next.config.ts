import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: '/logs/:path*',
        destination: '/api/logs/:path*',
      },
    ];
  },
};

export default nextConfig;
