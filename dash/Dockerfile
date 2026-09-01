# dash.lost.plus — the Grimoire dashboard as its own static site.
#
# It holds no data of its own: every number on screen comes from the Grimoire
# gateway's /stats endpoints, which nginx proxies here so the page is
# same-origin and the API key never reaches the browser. See dash/README.md.

FROM node:22-bookworm-slim AS build

WORKDIR /src

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY . .

RUN npm run build


FROM nginx:1.27-alpine AS runtime

# Rendered at container start by the base image's envsubst entrypoint, so the
# API key is supplied as an environment variable rather than baked into a layer.
# The filter is essential: without it envsubst would also eat nginx's own $uri,
# $host and friends.
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
ENV NGINX_ENVSUBST_FILTER="^(GRIMOIRE_API_KEY|GRIMOIRE_GATEWAY_URL)$"
ENV GRIMOIRE_GATEWAY_URL="http://grimoire:9001"

COPY security-headers.conf /etc/nginx/security-headers.conf
COPY --from=build /src/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
	CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1
