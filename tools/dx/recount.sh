#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Re-run the GitHub issue search counts behind docs/PAIN_POINTS.md and print then and now.
#   usage: tools/dx/recount.sh [evidence/dx/pain-points/counts.json]
# Uses GitHub's public search API (unauthenticated: about 10 searches a minute, so it waits 7 s between
# them). Entries that were not GitHub searches (Stack Overflow, Hacker News) are listed, not rerun.
F=${1:-evidence/dx/pain-points/counts.json}
python3 - "$F" <<'PY'
import json, sys, time, urllib.request, urllib.parse
for r in json.load(open(sys.argv[1])):
    api = r.get('api') or ''
    if 'api.github.com' not in api and not api.startswith('search/issues'):
        if r['query'].startswith('GitHub search/issues q='):
            q = r['query'].split('q=', 1)[1]
            api = 'https://api.github.com/search/issues?per_page=1&q=' + urllib.parse.quote(q)
        else:
            print(f"{r.get('total')!s:>8}  (not rerun)  {r['query']}"); continue
    if api.startswith('search/'): api = 'https://api.github.com/' + api
    try:
        with urllib.request.urlopen(urllib.request.Request(api, headers={'Accept': 'application/vnd.github+json'}), timeout=30) as h:
            now = json.load(h).get('total_count')
    except Exception as e:
        now = f'error {e}'
    print(f"{r.get('total')!s:>8} -> {now!s:<8} {r['query']}   (then: {r.get('run_at')})")
    time.sleep(7)
PY
