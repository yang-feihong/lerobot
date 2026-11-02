import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


PLAN = Path(__file__).resolve().parents[2] / "training_plan"
spec = importlib.util.spec_from_file_location("wandb_proxy", PLAN / "wandb_proxy.py")
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


class ProxyTests(unittest.TestCase):
    def test_private_forward_and_liveness(self):
        command = proxy.tunnel_command(Path("/config"), "wholebody_5", 18000, 18001)
        self.assertIn("127.0.0.1:18001:127.0.0.1:18000", command)
        for option in ("ExitOnForwardFailure=yes", "ServerAliveInterval=15", "ControlMaster=no"):
            self.assertIn(option, command)

    def test_unit_identity_separates_hosts_and_ports(self):
        units = {proxy.unit_name(Path("/config"), host, port, 18001)
                 for host in ("wholebody_F", "wholebody_1") for port in (18000, 18002)}
        self.assertEqual(len(units), 4)

    @patch.object(proxy, "ensure_user_manager")
    @patch.object(proxy.socket, "create_connection")
    @patch.object(proxy.subprocess, "run")
    def test_public_listener_stops_tunnel(self, run, connect, manager):
        run.side_effect = [subprocess.CompletedProcess([], 0, stdout="loaded\n"),
                           subprocess.CompletedProcess([], 0),
                           subprocess.CompletedProcess([], 77),
                           subprocess.CompletedProcess([], 0)]
        with self.assertRaisesRegex(RuntimeError, "not loopback-only"):
            proxy.ensure(Path("/config"), "wholebody_5", 18000, 18001)
        self.assertEqual(run.call_args.args[0][:3], ["systemctl", "--user", "stop"])

    @patch.object(proxy.time, "sleep")
    @patch.object(proxy, "ensure_user_manager")
    @patch.object(proxy.socket, "create_connection")
    @patch.object(proxy.subprocess, "run")
    def test_existing_service_is_reused_but_probed(self, run, connect, manager, sleep):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="loaded\n")
        proxy.ensure(Path("/config"), "wholebody_5", 18000, 18001)
        self.assertEqual(run.call_count, 3)
        self.assertEqual(run.call_args_list[-1].args[0][0], "ssh")
        self.assertNotIn("systemd-run", str(run.call_args_list))

    @patch.object(proxy.time, "sleep")
    @patch.object(proxy, "ensure_user_manager")
    @patch.object(proxy.socket, "create_connection")
    @patch.object(proxy.subprocess, "run")
    def test_service_restarts_and_failed_probe_blocks_training(self, run, connect, manager, sleep):
        run.return_value = subprocess.CompletedProcess([], 1, stdout="not-found\n")
        with self.assertRaisesRegex(RuntimeError, "preflight failed"):
            proxy.ensure(Path("/config"), "wholebody_5", 18000, 18001)
        service = run.call_args_list[1].args[0]
        self.assertIn("--property=Restart=always", service)
        self.assertIn("--property=RestartSec=5", service)

    @patch.object(proxy.socket, "create_connection", side_effect=OSError("proxy unavailable"))
    @patch.object(proxy.subprocess, "run")
    def test_missing_local_proxy_fails_before_remote_changes(self, run, connect):
        with self.assertRaises(OSError):
            proxy.ensure(Path("/config"), "wholebody_5", 18000, 18001)
        run.assert_not_called()

    def test_start_resume_share_network_hook(self):
        source = (PLAN / "cluster.sh").read_text()
        section = source.split("run_task_container() {", 1)[1].split("run_task_dry_run()", 1)[0]
        self.assertIn('ensure_wandb_proxy "${node}"', section)
        self.assertIn("--network host", section)
        self.assertIn("-e HTTPS_PROXY=", section)
        self.assertIn('code="${7:-$(container_code_dir)}"', section)

    @patch.object(proxy.subprocess, "run")
    def test_user_manager_survives_logout(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="no\n")
        proxy.ensure_user_manager()
        self.assertEqual(run.call_args.args[0][:2], ["loginctl", "enable-linger"])

    @patch.object(proxy.subprocess, "run")
    def test_lingering_already_enabled_is_unchanged(self, run):
        run.return_value = subprocess.CompletedProcess([], 0, stdout="yes\n")
        proxy.ensure_user_manager()
        self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
