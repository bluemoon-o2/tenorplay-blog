---
title: {{ .Title }}
date: {{ .Date.Format "2006-01-02" }}
tags: [{{ range $i, $e := .Params.tags }}{{ if $i }}, {{ end }}{{ $e }}{{ end }}]
url: {{ .Permalink }}
---

{{ .RawContent }}
