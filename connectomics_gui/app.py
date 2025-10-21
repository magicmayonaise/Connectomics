"""Interactive GUI for exploring FAFB synaptic partners using fafbseg data."""

from __future__ import annotations

import importlib.util
import logging
import threading
import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox, ttk
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DATASTACK = "flywire_fafb_production"
DEFAULT_SYNAPSE_TABLE = "synapses_nt_v1"
MAX_DISPLAYED_PARTNERS = 50

CAVECLIENT_SPEC = importlib.util.find_spec("caveclient")

if CAVECLIENT_SPEC is not None:
    from caveclient import CAVEclient
else:  # pragma: no cover - imported dynamically when available
    CAVEclient = None  # type: ignore[assignment]


@dataclass(frozen=True)
class SynapsePartner:
    """Summary of synaptic connectivity to a partner neuron."""

    partner_root_id: int
    synapse_count: int
    fraction: float

    @property
    def percentage(self) -> float:
        """Percentage representation of the partner's synapse count."""

        return self.fraction * 100.0


class SynapseFetcher:
    """Fetches synapse partners for a neuron from a FAFB materialization."""

    def __init__(self, datastack: str = DEFAULT_DATASTACK, synapse_table: str = DEFAULT_SYNAPSE_TABLE) -> None:
        self.datastack = datastack
        self.synapse_table = synapse_table
        self._client: Optional[Any] = None

    def fetch_partners(self, root_id: int, materialization_version: int) -> Tuple[List[SynapsePartner], List[SynapsePartner]]:
        """Return outgoing and incoming partner summaries for ``root_id``."""

        client = self._get_client()
        outgoing = self._query_synapses(client, {"pre_pt_root_id": root_id}, materialization_version)
        incoming = self._query_synapses(client, {"post_pt_root_id": root_id}, materialization_version)

        outgoing_summary = self._summarize_partners(outgoing, partner_field="post_pt_root_id")
        incoming_summary = self._summarize_partners(incoming, partner_field="pre_pt_root_id")
        return outgoing_summary, incoming_summary

    def get_latest_materialization(self) -> int:
        """Return the latest materialization version for the configured datastack."""

        client = self._get_client()
        materialize = getattr(client, "materialize", None)
        if materialize is None:
            raise RuntimeError("Connected client does not expose a materialize interface.")
        getter = getattr(materialize, "get_latest_version", None)
        if getter is None:
            raise RuntimeError("Materialize client does not support querying the latest version.")
        return int(getter())

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def _create_client(self) -> Any:
        if CAVECLIENT_SPEC is None:
            raise RuntimeError(
                "caveclient is not available. Install fafbseg or caveclient to enable partner queries."
            )
        return CAVEclient(self.datastack)

    def _query_synapses(
        self, client: Any, filter_dict: Mapping[str, Any], materialization_version: int
    ) -> Iterable[Mapping[str, Any]]:
        materialize = getattr(client, "materialize", None)
        if materialize is None:
            raise RuntimeError("Connected client does not expose a materialize interface.")
        query_fn = getattr(materialize, "query_table", None)
        if query_fn is None:
            raise RuntimeError("Materialize interface does not provide query_table().")
        logger.debug(
            "Querying synapses table %s with filter %s at materialization %s", self.synapse_table, filter_dict, materialization_version
        )
        result = query_fn(  # type: ignore[call-arg]
            self.synapse_table,
            materialization_version=materialization_version,
            filter_in_dict=dict(filter_dict),
        )
        return self._normalize_records(result)

    def _normalize_records(self, result: Any) -> List[Mapping[str, Any]]:
        if result is None:
            return []

        if isinstance(result, list):
            normalized: List[Mapping[str, Any]] = []
            for record in result:
                if isinstance(record, Mapping):
                    normalized.append(record)
                elif hasattr(record, "_asdict"):
                    normalized.append(record._asdict())
                else:
                    normalized.append(dict(record))
            return normalized

        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            try:
                converted = to_dict("records")
            except TypeError:
                converted = to_dict()
            if isinstance(converted, list):
                return [record if isinstance(record, Mapping) else dict(record) for record in converted]
            if isinstance(converted, Mapping):
                keys = list(converted.keys())
                length = len(converted[keys[0]]) if keys else 0
                normalized_list = []
                for index in range(length):
                    normalized_list.append({key: converted[key][index] for key in keys})
                return normalized_list

        if isinstance(result, Sequence):
            return [record if isinstance(record, Mapping) else dict(record) for record in result]

        raise TypeError(f"Unsupported synapse query result type: {type(result)!r}")

    def _summarize_partners(self, records: Iterable[Mapping[str, Any]], partner_field: str) -> List[SynapsePartner]:
        counts: Dict[int, int] = {}
        total = 0
        for record in records:
            partner = record.get(partner_field)
            if partner is None:
                continue
            try:
                partner_id = int(partner)
            except (TypeError, ValueError):
                logger.debug("Skipping partner entry with non-integer id: %r", partner)
                continue
            counts[partner_id] = counts.get(partner_id, 0) + 1
            total += 1

        if total == 0:
            return []

        partners = [
            SynapsePartner(partner_root_id=partner_id, synapse_count=count, fraction=count / total)
            for partner_id, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
        ]
        return partners


class ConnectomicsApp(tk.Tk):
    """Tkinter application that wraps synapse partner queries in a GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("FAFBseg Synaptic Partner Explorer")
        self.geometry("960x640")
        self.minsize(720, 480)

        self.root_id_var = tk.StringVar()
        self.materialization_var = tk.StringVar()
        self.datastack_var = tk.StringVar(value=DEFAULT_DATASTACK)
        self.postsyn_filter_var = tk.StringVar()
        self.status_var = tk.StringVar(value=self._initial_status_message())

        self._postsyn_filter_trace = self.postsyn_filter_var.trace_add(
            "write", lambda *_: self._on_postsyn_filter_change()
        )

        self._current_fetch_thread: Optional[threading.Thread] = None
        self._postsyn_data: List[SynapsePartner] = []
        self._presyn_data: List[SynapsePartner] = []

        self._build_layout()
        self.bind("<Return>", lambda event: self.on_fetch())

    def _initial_status_message(self) -> str:
        if CAVECLIENT_SPEC is None:
            return "caveclient is not installed. Install fafbseg to enable partner queries."
        return "Enter a root ID and materialization version, then press Fetch partners."

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=(16, 16))
        container.pack(fill=tk.BOTH, expand=True)

        form = ttk.Frame(container)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Root ID").grid(row=0, column=0, sticky="w")
        root_entry = ttk.Entry(form, textvariable=self.root_id_var, width=20)
        root_entry.grid(row=1, column=0, sticky="ew", padx=(0, 12))
        root_entry.focus()

        ttk.Label(form, text="Materialization version").grid(row=0, column=1, sticky="w")
        materialization_entry = ttk.Entry(form, textvariable=self.materialization_var, width=16)
        materialization_entry.grid(row=1, column=1, sticky="ew", padx=(0, 12))

        ttk.Label(form, text="Datastack").grid(row=0, column=2, sticky="w")
        datastack_entry = ttk.Entry(form, textvariable=self.datastack_var, width=28)
        datastack_entry.grid(row=1, column=2, sticky="ew", padx=(0, 12))

        self.fetch_button = ttk.Button(form, text="Fetch partners", command=self.on_fetch)
        self.fetch_button.grid(row=1, column=3, sticky="ew")

        for column in range(3):
            form.columnconfigure(column, weight=1)
        form.columnconfigure(3, weight=0)

        status_label = ttk.Label(container, textvariable=self.status_var, foreground="#555555")
        status_label.pack(fill=tk.X, pady=(12, 8))

        notebook = ttk.Notebook(container)
        notebook.pack(fill=tk.BOTH, expand=True)

        self.postsyn_frame = ttk.Frame(notebook)
        notebook.add(self.postsyn_frame, text="Postsynaptic partners (outgoing)")
        self.presyn_frame = ttk.Frame(notebook)
        notebook.add(self.presyn_frame, text="Presynaptic partners (incoming)")

        self.postsyn_tree, self.postsyn_total_var = self._build_postsyn_section(self.postsyn_frame)
        self.presyn_tree, self.presyn_total_var = self._build_presyn_section(self.presyn_frame)

        if CAVECLIENT_SPEC is None:
            self.fetch_button.state(["disabled"])

    def _build_postsyn_section(self, parent: ttk.Frame) -> Tuple[ttk.Treeview, tk.StringVar]:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        filter_frame = ttk.Frame(parent)
        filter_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        filter_frame.columnconfigure(1, weight=1)

        ttk.Label(filter_frame, text="Filter partners").grid(row=0, column=0, sticky="w", padx=(0, 8))
        filter_entry = ttk.Entry(filter_frame, textvariable=self.postsyn_filter_var)
        filter_entry.grid(row=0, column=1, sticky="ew")
        filter_entry.configure(takefocus=True)

        table_frame = ttk.Frame(parent)
        table_frame.grid(row=1, column=0, sticky="nsew")

        return self._build_partner_table(table_frame)

    def _build_presyn_section(self, parent: ttk.Frame) -> Tuple[ttk.Treeview, tk.StringVar]:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        return self._build_partner_table(parent)

    def _build_partner_table(self, parent: ttk.Frame) -> Tuple[ttk.Treeview, tk.StringVar]:
        columns = ("partner", "synapses", "percentage")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=15)
        tree.heading("partner", text="Partner root ID")
        tree.heading("synapses", text="Synapses")
        tree.heading("percentage", text="Percentage")

        tree.column("partner", width=220, anchor=tk.W)
        tree.column("synapses", width=120, anchor=tk.CENTER)
        tree.column("percentage", width=120, anchor=tk.CENTER)

        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        total_var = tk.StringVar(value="Synapses: 0")
        total_label = ttk.Label(parent, textvariable=total_var)
        total_label.grid(row=1, column=0, sticky="w", pady=(8, 0))

        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        return tree, total_var

    def on_fetch(self) -> None:
        if CAVECLIENT_SPEC is None:
            messagebox.showerror("Missing dependency", "caveclient is required. Install fafbseg to continue.")
            return

        if self._current_fetch_thread and self._current_fetch_thread.is_alive():
            messagebox.showwarning("Fetch in progress", "Please wait for the current fetch to finish.")
            return

        root_id_value = self.root_id_var.get().strip()
        materialization_value = self.materialization_var.get().strip()
        datastack_value = self.datastack_var.get().strip() or DEFAULT_DATASTACK

        if not root_id_value:
            messagebox.showerror("Missing input", "Please enter a root ID to search.")
            return

        if not materialization_value:
            messagebox.showerror("Missing input", "Please provide a materialization version.")
            return

        try:
            root_id = int(root_id_value)
        except ValueError:
            messagebox.showerror("Invalid root ID", "Root ID must be an integer.")
            return

        try:
            materialization_version = int(materialization_value)
        except ValueError:
            messagebox.showerror("Invalid materialization version", "Materialization version must be an integer.")
            return

        self.status_var.set("Fetching synaptic partners...")
        self.fetch_button.state(["disabled"])
        self._clear_tables()

        thread = threading.Thread(
            target=self._run_fetch,
            args=(root_id, materialization_version, datastack_value),
            daemon=True,
        )
        thread.start()
        self._current_fetch_thread = thread

    def _run_fetch(self, root_id: int, materialization_version: int, datastack: str) -> None:
        fetcher = SynapseFetcher(datastack=datastack)
        try:
            postsynaptic, presynaptic = fetcher.fetch_partners(root_id, materialization_version)
        except Exception as exc:  # pragma: no cover - GUI error handling path
            logger.exception("Failed to fetch synapse partners")
            self.after(0, lambda: self._handle_fetch_error(exc))
            return

        self.after(
            0,
            lambda: self._handle_fetch_success(
                root_id=root_id,
                materialization_version=materialization_version,
                datastack=datastack,
                postsynaptic=postsynaptic,
                presynaptic=presynaptic,
            ),
        )

    def _clear_tables(self) -> None:
        self._postsyn_data = []
        self._presyn_data = []

        for tree, total_var in (
            (self.postsyn_tree, self.postsyn_total_var),
            (self.presyn_tree, self.presyn_total_var),
        ):
            for item in tree.get_children():
                tree.delete(item)
            total_var.set("Synapses: 0")

    def _handle_fetch_error(self, error: Exception) -> None:
        self.fetch_button.state(["!disabled"])
        message = str(error) or error.__class__.__name__
        self.status_var.set(f"Failed to fetch partners: {message}")
        messagebox.showerror("Fetch failed", message)

    def _handle_fetch_success(
        self,
        *,
        root_id: int,
        materialization_version: int,
        datastack: str,
        postsynaptic: Sequence[SynapsePartner],
        presynaptic: Sequence[SynapsePartner],
    ) -> None:
        self.fetch_button.state(["!disabled"])

        self._postsyn_data = list(postsynaptic)
        self._presyn_data = list(presynaptic)

        self._refresh_postsyn_table()
        self._refresh_presyn_table()

        if postsynaptic or presynaptic:
            self.status_var.set(
                f"Loaded partners for root {root_id} (materialization {materialization_version}, datastack {datastack})."
            )
        else:
            self.status_var.set(
                f"No synapses found for root {root_id} at materialization {materialization_version}."
            )

    def _refresh_postsyn_table(self) -> None:
        filter_value = self.postsyn_filter_var.get().strip()
        if filter_value:
            normalized_terms = [
                term
                for term in filter_value.replace(",", " ").split()
                if term
            ]
            if normalized_terms:
                filtered = [
                    partner
                    for partner in self._postsyn_data
                    if any(term in str(partner.partner_root_id) for term in normalized_terms)
                ]
            else:
                filtered = list(self._postsyn_data)
        else:
            filtered = list(self._postsyn_data)

        displayed_count = self._populate_table(self.postsyn_tree, filtered)
        self.postsyn_total_var.set(
            self._format_summary(
                self._postsyn_data,
                filtered_count=len(filtered),
                displayed_count=displayed_count,
            )
        )

    def _refresh_presyn_table(self) -> None:
        displayed_count = self._populate_table(self.presyn_tree, self._presyn_data)
        self.presyn_total_var.set(
            self._format_summary(self._presyn_data, displayed_count=displayed_count)
        )

    def _populate_table(self, tree: ttk.Treeview, partners: Sequence[SynapsePartner]) -> int:
        for item in tree.get_children():
            tree.delete(item)

        displayed_partners = partners[:MAX_DISPLAYED_PARTNERS]
        for partner in displayed_partners:
            tree.insert(
                "",
                tk.END,
                values=(partner.partner_root_id, partner.synapse_count, f"{partner.percentage:.2f}%"),
            )

        return len(displayed_partners)

    def _on_postsyn_filter_change(self) -> None:
        if not self._postsyn_data:
            return
        self._refresh_postsyn_table()

    def _format_summary(
        self,
        partners: Sequence[SynapsePartner],
        *,
        filtered_count: Optional[int] = None,
        displayed_count: Optional[int] = None,
    ) -> str:
        total_synapses = sum(partner.synapse_count for partner in partners)
        if not partners:
            return "Synapses: 0"
        partner_count = len(partners)
        summary = f"Synapses: {total_synapses} across {partner_count} partner"
        if partner_count != 1:
            summary += "s"

        if filtered_count is not None and filtered_count != partner_count:
            summary += f" (filtered to {filtered_count})"

        effective_filtered = filtered_count if filtered_count is not None else partner_count
        if displayed_count is not None and effective_filtered and displayed_count < effective_filtered:
            summary += f" — showing top {displayed_count}"

        return summary


def main() -> None:
    """Entry point that launches the Tkinter GUI."""

    logging.basicConfig(level=logging.INFO)
    app = ConnectomicsApp()
    app.mainloop()


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    main()
