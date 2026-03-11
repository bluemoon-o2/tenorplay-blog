const CONFIG = {
  ALLOWED_REFERERS: new Set([
    'tensorplay.cn',
    'blog.tensorplay.cn'
  ]),
  IMAGEKIT_BASE_URL: 'https://ik.imagekit.io/nwbaln7ps',
  FAVICON_URL: 'https://blog.tensorplay.cn/images/avator.png',
  // 水印参数（抽离为常量，便于修改）
  IMAGEKIT_WATERMARK_PARAMS: 'tr=l-text,ie-VGVuc29yUGxheSBCbG9n,lx-bw_mul_0.7,ly-bh_mul_0.9,lm-cutout,co-FFFFFF50,tg-i,fs-bh_div_15,ff-Amaranth,l-end',
  ALLOWED_METHODS: new Set(['GET', 'HEAD']),
  CORS_ALLOWED_ORIGINS: [
    'https://tensorplay.cn',
    'https://blog.tensorplay.cn'
  ]
};

function validateReferer(referer) {
  if (!referer) return false;

  try {
    const refererUrl = new URL(referer);
    const hostname = refererUrl.hostname.toLowerCase();

    return Array.from(CONFIG.ALLOWED_REFERERS).some(allowedDomain => {
      return hostname === allowedDomain || (hostname.endsWith(`.${allowedDomain}`) && !hostname.startsWith('.'));
    });
  } catch (e) {
    console.error('Referer 解析失败:', e);
    return false;
  }
}

function processKey(pathname) {
  let key = pathname.slice(1).replace(/^blog\//, '');
  key = decodeURIComponent(key);
  key = key.replace(/(\.\.\/|\.\/)/g, '');
  return key.trim();
}

function getCorsOrigin(referer) {
  if (!referer) return '';
  try {
    const refererUrl = new URL(referer);
    const origin = `${refererUrl.protocol}//${refererUrl.host}`;
    return CONFIG.CORS_ALLOWED_ORIGINS.includes(origin) ? origin : '';
  } catch (e) {
    return '';
  }
}

export default {
  async fetch(request, env) {
    try {
      const url = new URL(request.url);
      const method = request.method.toUpperCase();

      // 0. 处理 Favicon
      if (url.pathname === '/favicon.ico') {
        return Response.redirect(CONFIG.FAVICON_URL, 301);
      }

      // 1. 验证请求方法
      if (!CONFIG.ALLOWED_METHODS.has(method)) {
        return new Response('Method Not Allowed', { status: 405, headers: { 'Allow': Array.from(CONFIG.ALLOWED_METHODS).join(', ') } });
      }

      const referer = request.headers.get('Referer');
      if (!validateReferer(referer)) {
        return new Response('Forbidden: Invalid Referer', { status: 403 });
      }

      const key = processKey(url.pathname);
      if (!key) {
        return new Response('TensorPlay Image Server', { status: 200, headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
      }

      const imageKitUrl = `${CONFIG.IMAGEKIT_BASE_URL}/${encodeURIComponent(key)}?${CONFIG.IMAGEKIT_WATERMARK_PARAMS}`;

      const imageKitResponse = await fetch(imageKitUrl, {
        method: method,
        headers: { 'User-Agent': 'Cloudflare-Workers/TensorPlay' }
      });

      const newHeaders = new Headers(imageKitResponse.headers);
      const corsOrigin = getCorsOrigin(referer);
      if (corsOrigin) {
        newHeaders.set('Access-Control-Allow-Origin', corsOrigin);
        newHeaders.set('Access-Control-Allow-Methods', Array.from(CONFIG.ALLOWED_METHODS).join(', '));
        newHeaders.set('Access-Control-Allow-Headers', 'Origin, Referer, Accept');
      }
      newHeaders.delete('x-imagekit-id');
      newHeaders.set('Cache-Control', 'public, max-age=86400, s-maxage=31536000');

      return new Response(imageKitResponse.body, {
        status: imageKitResponse.status,
        statusText: imageKitResponse.statusText,
        headers: newHeaders
      });

    } catch (error) {
      return new Response('Internal Server Error', { status: 500 });
    }
  },
};
