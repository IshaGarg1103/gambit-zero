"""Continue an interrupted HTTP body within ONE live decoder/hash session.

Ranges extend the exact original byte stream; they are never independently
Zstandard-decoded. Process exit still requires a month restart from byte zero.
"""
import http.client
import re
import ssl
import time
import urllib.error
import urllib.request

from tinychess.data.archive_scan import ScanError


class ContinuousArchiveResponse:
    def __init__(self, source, *, opener=urllib.request.urlopen, max_seconds,
                 max_reconnects=8, timeout=60):
        self.source, self.opener = source, opener
        self.expected = source['archive_size_bytes']
        self.etag = source.get('headers', {}).get('ETag')
        self.deadline = time.monotonic() + max_seconds
        self.max_reconnects, self.timeout = max_reconnects, timeout
        self.offset = self.reconnects = 0
        self.response = None
        self.events = []

    def _remaining(self):
        remaining = self.deadline-time.monotonic()
        if remaining <= 0:
            raise ScanError('Runtime budget exhausted during archive connection recovery')
        return remaining

    def close(self):
        if self.response is not None:
            response, self.response = self.response, None
            response.close()

    def _recover(self, reason):
        self.close()
        self._remaining()
        if self.reconnects >= self.max_reconnects:
            raise ScanError('Archive reconnection limit exhausted; no verified month checkpoint')
        self.reconnects += 1
        self.events.append(dict(offset=self.offset, reason=reason))
        delay = min(2**(self.reconnects-1), 30)
        if delay >= self._remaining():
            raise ScanError('Runtime budget exhausted before archive reconnection')
        time.sleep(delay)

    def _open(self):
        headers = {'User-Agent': 'GambitZero-M1/1.0', 'Accept-Encoding': 'identity'}
        if self.offset:
            if not self.etag or self.etag.startswith('W/'):
                raise ScanError('Strong unchanged archive ETag required for HTTP continuation')
            headers.update(Range=f'bytes={self.offset}-', **{'If-Range': self.etag})
        request = urllib.request.Request(self.source['url'], headers=headers)
        response = self.opener(request, timeout=min(self.timeout, self._remaining()))
        try:
            expected_status = 206 if self.offset else 200
            if response.status != expected_status:
                raise ScanError(f'Archive HTTP continuation requires status {expected_status}; refusing byte restart')
            if response.headers.get('Content-Encoding', 'identity').lower() != 'identity':
                raise ScanError('Transformed archive HTTP encoding refused')
            try:
                length = int(response.headers.get('Content-Length', -1))
            except (TypeError, ValueError):
                raise ScanError('Invalid archive response length') from None
            if length != self.expected-self.offset:
                raise ScanError('Archive response length differs from frozen remaining bytes')
            etag = response.headers.get('ETag')
            if self.etag and etag != self.etag:
                raise ScanError('Archive ETag changed; refusing mixed-source stream')
            content_range = response.headers.get('Content-Range')
            if self.offset:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range or '')
                if not match or tuple(map(int, match.groups())) != (self.offset, self.expected-1, self.expected):
                    raise ScanError('Archive Content-Range is not the exact contiguous frozen suffix')
            elif content_range:
                raise ScanError('Initial full streaming scan requires HTTP 200 from byte zero')
            if self.etag is None:
                self.etag = etag
            self.response = response
        except BaseException:
            response.close()
            raise

    def read(self, size):
        if not 0 < size <= 65536:
            raise ValueError('Archive transport reads must be bounded to 1..65536 bytes')
        while True:
            self._remaining()
            partial = False
            try:
                if self.response is None:
                    self._open()
                chunk = self.response.read(size)
            except http.client.IncompleteRead as exc:
                chunk = exc.partial
                self.close()
                if chunk:
                    partial = True
                else:
                    self._recover('IncompleteRead')
                    continue
            except urllib.error.HTTPError as exc:
                if exc.code not in [408, 429, 500, 502, 503, 504]:
                    raise ScanError(f'Archive HTTP failure ({exc.code}); not a resumable response') from None
                self._recover('HTTP '+str(exc.code))
                continue
            except (TimeoutError, ConnectionError, urllib.error.URLError, ssl.SSLEOFError) as exc:
                if self.offset == self.expected:
                    raise ScanError('Unable to confirm actual transport EOF after frozen archive length') from None
                self._recover(type(exc).__name__)
                continue
            if not chunk and self.offset < self.expected:
                self._recover('premature_eof')
                continue
            if len(chunk) > size:
                raise ScanError('Archive transport returned more bytes than requested')
            self.offset += len(chunk)
            if partial and self.offset < self.expected:
                # Deliver received bytes once, then reconnect on the next read.
                # Decoder, PGN parser, sampler and SHA256 remain the same objects.
                self._recover('IncompleteRead_with_partial_bytes')
            return chunk

    def __enter__(self):
        # Open lazily in read(), so even connection-setup timeouts are bounded
        # by the same retry/deadline policy as body interruptions.
        return self

    def __exit__(self, *exc):
        self.close()
