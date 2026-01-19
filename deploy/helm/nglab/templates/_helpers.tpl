{{/*
Expand the name of the chart.
*/}}
{{- define "nglab.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "nglab.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "nglab.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "nglab.labels" -}}
helm.sh/chart: {{ include "nglab.chart" . }}
{{ include "nglab.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "nglab.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nglab.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "nglab.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "nglab.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Get the PostgreSQL host
*/}}
{{- define "nglab.postgresHost" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s-postgresql" (include "nglab.fullname" .) }}
{{- else }}
{{- .Values.externalDatabase.host }}
{{- end }}
{{- end }}

{{/*
Get the Redis host
*/}}
{{- define "nglab.redisHost" -}}
{{- if .Values.redis.enabled }}
{{- printf "%s-redis-master" (include "nglab.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.host }}
{{- end }}
{{- end }}

{{/*
Get the Jaeger endpoint
*/}}
{{- define "nglab.jaegerEndpoint" -}}
{{- if .Values.jaeger.enabled }}
{{- printf "http://%s-jaeger-collector:4317" (include "nglab.fullname" .) }}
{{- else }}
{{- .Values.externalJaeger.endpoint }}
{{- end }}
{{- end }}
