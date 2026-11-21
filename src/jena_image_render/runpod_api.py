"""Minimal RunPod GraphQL client: create with GPU fallback, inspect, terminate."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import requests

ENDPOINT = "https://api.runpod.io/graphql"
CREATE = """mutation Create($input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: $input) { id machine { gpuDisplayName } } }"""
GET = """query Pod($input: PodFilter!) { pod(input: $input) {
  id desiredStatus lastStatusChange machine { podHostId }
  runtime { uptimeInSeconds container { cpuPercent memoryPercent } } } }"""
TERMINATE = "mutation Stop($input: PodTerminateInput!) { podTerminate(input: $input) }"
ACCOUNT = """query { myself { clientBalance currentSpendPerHr
  pods { id name desiredStatus } } }"""
CAPACITY_ERRORS = ("instances available", "no longer any", "not enough")
PORT = 3000


class NoCapacity(Exception):
    """None of the requested GPU types could be rented."""


def proxy_url(pod_id: str, port: int = PORT) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net"


class RunPodAPI:
    def __init__(self, key: str, post: Callable[..., Any] = requests.post) -> None:
        if not key:
            raise ValueError("RUNPOD_API_KEY is not set (put it in .env)")
        self.key = key
        self.post = post

    def _call(self, query: str, variables: dict[str, Any] | None = None) -> Any:
        response = self.post(
            ENDPOINT,
            json={"query": query, "variables": variables or {}},
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=60,
        )
        payload = response.json()
        if payload.get("errors"):
            raise RuntimeError(
                "; ".join(e.get("message", "") for e in payload["errors"])
            )
        if response.status_code >= 400:
            raise RuntimeError(f"RunPod API HTTP {response.status_code}")
        return payload["data"]

    def create_pod(
        self,
        name: str,
        image: str,
        env: dict[str, str],
        gpu_types: Sequence[str],
        secure: bool,
        host_memory_gb: Sequence[int],
        cuda_versions: Sequence[str],
        disk_gb: int = 100,
    ) -> dict[str, str]:
        """Try each GPU type in order; only capacity errors move to the next one.

        The whole GPU list is tried per host-RAM tier, roomiest tier first.
        """
        errors = []
        for memory in host_memory_gb:
            for gpu in gpu_types:
                spec = {
                    "cloudType": "SECURE" if secure else "COMMUNITY",
                    "gpuCount": 1,
                    "gpuTypeId": gpu,
                    "name": name,
                    "imageName": image,
                    "containerDiskInGb": disk_gb,
                    "volumeInGb": 0,
                    "minMemoryInGb": memory,
                    "minVcpuCount": 4,
                    "allowedCudaVersions": list(cuda_versions),
                    "ports": f"{PORT}/http",
                    "env": [{"key": k, "value": v} for k, v in env.items()],
                }
                try:
                    pod = self._call(CREATE, {"input": spec})["podFindAndDeployOnDemand"]
                except RuntimeError as exc:
                    if any(text in str(exc).lower() for text in CAPACITY_ERRORS):
                        errors.append(f"{gpu}@{memory}GB")
                        continue
                    raise
                machine = pod.get("machine") or {}
                return {
                    "id": pod["id"],
                    "gpu": machine.get("gpuDisplayName") or gpu,
                    "host_memory_gb": str(memory),
                }
        raise NoCapacity("no stock for " + ", ".join(errors))

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        pod: dict[str, Any] | None = self._call(GET, {"input": {"podId": pod_id}})["pod"]
        return pod

    def terminate(self, pod_id: str) -> None:
        self._call(TERMINATE, {"input": {"podId": pod_id}})

    def account(self) -> dict[str, Any]:
        myself: dict[str, Any] = self._call(ACCOUNT)["myself"]
        return myself
