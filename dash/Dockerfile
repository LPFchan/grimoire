# dash.lost.plus — the Grimoire dashboard as its own static site.
#
# It holds no data of its own: every number on screen comes from the Grimoire
# gateway's /stats endpoints, fetched from the browser. See dash/README.md.

FROM node:22-bookworm-slim AS build

WORKDIR /src

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .

# Baked in at build time as the default gateway; overridable per-browser from
# the sign-in screen.
ARG VITE_API_BASE=https://chat.lost.plus
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build


FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /src/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
	CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
