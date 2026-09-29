# syntax=docker/dockerfile:1
#
# The web app as a container: the Next.js standalone server produced by
# `next build` (output: "standalone"), with the client bundle's WebSocket URL
# baked in as a build argument because NEXT_PUBLIC_* variables are inlined at
# build time.
#
# Build from the repository root so the shared workspace package is in the
# context:
#   docker build -f packages/web/Dockerfile --build-arg NEXT_PUBLIC_WS_URL=... \
#     -t open-inspect-web .

FROM node:24-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
COPY packages/shared/package.json packages/shared/
COPY packages/web/package.json packages/web/
# --ignore-scripts: the root package's prepare script installs git hooks
# through husky, which a workspace-scoped install does not bring along.
# eslint is a repository-root devDependency the scoped install omits, but
# next build type-checks the co-located tests that import it.
RUN --mount=type=cache,target=/root/.npm \
    npm ci --ignore-scripts --workspace=@open-inspect/shared --workspace=@open-inspect/web \
    && npm install --no-save --ignore-scripts eslint@^9.18.0
COPY packages/shared/ packages/shared/
COPY packages/web/ packages/web/
ARG NEXT_PUBLIC_WS_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL
RUN npm run build -w @open-inspect/shared \
    && npm run build -w @open-inspect/web

# outputFileTracingRoot is the monorepo root, so the standalone folder holds
# packages/web/server.js beside the traced node_modules at its own root.
FROM node:24-slim
ENV NODE_ENV=production \
    HOSTNAME=0.0.0.0 \
    PORT=3000
WORKDIR /app
COPY --from=build /app/packages/web/.next/standalone ./
COPY --from=build /app/packages/web/.next/static ./packages/web/.next/static
COPY --from=build /app/packages/web/public ./packages/web/public
USER node
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
    CMD ["node", "-e", "fetch('http://127.0.0.1:3000/').then((r) => process.exit(r.status < 400 ? 0 : 1), () => process.exit(1))"]
CMD ["node", "packages/web/server.js"]
