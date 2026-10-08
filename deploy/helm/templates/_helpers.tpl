{{- define "embedforge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "embedforge.fullname" -}}
{{- printf "%s" (include "embedforge.name" .) }}
{{- end }}

{{- define "embedforge.labels" -}}
app: {{ include "embedforge.name" . }}
{{- end }}

{{- define "embedforge.selectorLabels" -}}
app: {{ include "embedforge.name" . }}
{{- end }}
