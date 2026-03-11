const ALLOWED_REFERERS = [
  'tensorplay.cn',
  'blog.tensorplay.cn',
];

const R2_BUCKET_BINDING = 'blog'; 

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const referer = request.headers.get('Referer');

    if (referer) {
      try {
        const refererUrl = new URL(referer);
        const hostname = refererUrl.hostname;
        
        const isAllowed = ALLOWED_REFERERS.some(domain => 
          hostname === domain || hostname.endsWith(`.${domain}`)
        );

        if (!isAllowed) {
          return new Response('Forbidden: Access Denied', { status: 403 });
        }
      } catch (e) {
        return new Response('Invalid Referer', { status: 403 });
      }
    }

    let key = url.pathname.slice(1);
    key = decodeURIComponent(key);

    if (!key) {
      return new Response('TensorPlay Image Server', { status: 200 });
    }

    const bucket = env[R2_BUCKET_BINDING];
    if (!bucket) {
      return new Response(`Server Error: R2 Bucket '${R2_BUCKET_BINDING}' not bound`, { status: 500 });
    }

    let object = await bucket.get(key);

    if (object === null) {
      return new Response(`Image Not Found: ${key}`, { status: 404 });
    }

    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set('etag', object.httpEtag);
    
    headers.set('Cache-Control', 'public, max-age=31536000, immutable');
    headers.set('Access-Control-Allow-Origin', '*');

    return new Response(object.body, {
      headers,
    });
  },
};
