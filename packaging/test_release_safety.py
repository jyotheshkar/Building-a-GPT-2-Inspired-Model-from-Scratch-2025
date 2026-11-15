"""Test consequential release boundaries using tiny fixtures, without network."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_python
import publish_release as publisher
import release_common as common
import release_gate as gate

SHA = "a" * 40
OTHER_SHA = "b" * 40
NAME = "jyotheshkar/Building-a-GPT-2-Inspired-Model-from-Scratch-2025"


def report(frozen):
    return {"status":"passed", "version":common.CONFIG["version"], "frozen":frozen, "device":"cpu",
            "training_steps":3, "parameters_changed":True, "parameters_finite":True,
            "checkpoint_roundtrip":True, "tokenizer_roundtrip":True, "generation_length":17,
            "generation_finite":True, "causal_mask_passed":True,
            "gui":{"status":"passed", "widgets_constructed":True,
                   "default_source_loaded":True, "initial_idle_state":True,
                   "gui_training_steps":2, "gui_generation_length":9,
                   "gui_checkpoint_roundtrip":True, "gui_text_export":True}}


class ReleaseSafety(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.assets = Path(self.directory.name)
        environment = patch.dict(os.environ, {"GITHUB_EVENT_NAME":"push", "GITHUB_REF":"refs/heads/main",
            "GITHUB_SHA":SHA, "GITHUB_REPOSITORY":NAME, "GITHUB_REPOSITORY_ID":str(common.REPOSITORY_ID)})
        environment.start()
        self.addCleanup(environment.stop)
        for target, config in common.CONFIG["platforms"].items():
            archive = self.assets / config["archive"]
            archive.write_bytes(target.encode())
            common.write_json(self.assets/f"BUILD-MANIFEST-{target}.json", {
                "source_commit":SHA, "version":common.CONFIG["version"], "tag":common.CONFIG["tag"],
                "target":target, "extracted_archive_smoke":report(True),
                "archive":{"sha256":common.digest(archive), "bytes":archive.stat().st_size},
                "tcl_tk_runtime":{"tcl":{}, "tk":{}}})
        distributions=[]
        for key in ("wheel", "sdist"):
            path = self.assets / common.CONFIG["distribution"][key]
            path.write_bytes(key.encode())
            distributions.append({"filename":path.name, "sha256":common.digest(path), "bytes":path.stat().st_size})
        common.write_json(self.assets/"PYTHON-DISTRIBUTIONS.json", {
            "source_commit":SHA, "version":common.CONFIG["version"], "tag":common.CONFIG["tag"],
            "wheel_built_from_sdist":True, "historical_notebooks_included":False,
            "wheel_smoke":report(False), "distributions":distributions})

    def draft(self):
        paths = publisher.validate_assets(SHA, self.assets)
        return {"draft":True, "prerelease":False,
                "assets":[{"name":path.name,"state":"uploaded","size":path.stat().st_size,
                           "digest":"sha256:"+common.digest(path)} for path in paths]}

    def test_pr_never_queries_or_writes_release(self):
        with patch.dict(os.environ,{"GITHUB_EVENT_NAME":"pull_request"}), patch.object(gate,"verify_identity") as identity:
            self.assertTrue(gate.should_build(SHA))
            identity.assert_not_called()

    def test_wrong_repository_id_rejected_before_api(self):
        with patch.dict(os.environ,{"GITHUB_REPOSITORY_ID":"1"}), patch.object(common,"api") as api:
            with self.assertRaisesRegex(RuntimeError,"repository ID"):
                common.verify_identity()
            api.assert_not_called()

    def test_repository_rename_retains_identity(self):
        with patch.object(common,"api",return_value={"id":common.REPOSITORY_ID}) as api:
            self.assertEqual(common.verify_identity(),NAME)
            api.assert_called_once_with("repos/"+NAME)

    def test_existing_version_skips_later_main_commit_without_writes(self):
        with patch.object(gate,"verify_identity",return_value=NAME), \
                patch.object(gate,"tag_sha",return_value=OTHER_SHA), \
                patch.object(gate,"find_release",return_value={"id":42}) as find, \
                patch.object(gate,"verify_existing") as verify:
            self.assertFalse(gate.should_build(SHA))
            verify.assert_called_once_with(NAME,{"id":42},OTHER_SHA)
            find.assert_called_once()

    def test_orphan_tag_requires_inspection(self):
        with patch.object(gate,"verify_identity",return_value=NAME), \
                patch.object(gate,"tag_sha",return_value=SHA), patch.object(gate,"find_release",return_value=None):
            with self.assertRaisesRegex(RuntimeError,"tag already exists"):
                gate.should_build(SHA)

    def test_corrupt_archive_rejected_before_github_access(self):
        name = next(iter(common.CONFIG["platforms"].values()))["archive"]
        (self.assets/name).write_bytes(b"changed")
        with patch.object(publisher,"verify_identity") as identity:
            with self.assertRaisesRegex(RuntimeError,"evidence failed"):
                publisher.publish(SHA,self.assets)
            identity.assert_not_called()

    def test_other_source_tag_is_never_overwritten(self):
        with patch.object(publisher,"verify_identity",return_value=NAME), \
                patch.object(publisher,"tag_sha",return_value=OTHER_SHA), \
                patch.object(publisher,"find_release",return_value=None), \
                patch.object(publisher.subprocess,"run") as run:
            with self.assertRaisesRegex(RuntimeError,"another source"):
                publisher.publish(SHA,self.assets)
            run.assert_not_called()

    def test_exact_tag_precedes_draft_then_verified_publication(self):
        draft = self.draft()
        published = {**draft,"draft":False}
        with patch.object(publisher,"verify_identity",return_value=NAME), \
                patch.object(publisher,"tag_sha",side_effect=[None,SHA,SHA]), \
                patch.object(publisher,"find_release",side_effect=[None,draft,published]), \
                patch.object(publisher,"verify_existing") as verify, \
                patch.object(publisher.subprocess,"run") as run:
            publisher.publish(SHA,self.assets)
            verify.assert_called_once_with(NAME,published,SHA)
        commands=[call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:3],["gh","api",f"repos/{NAME}/git/refs"])
        self.assertIn("sha="+SHA,commands[0])
        self.assertEqual(commands[1][:3],["gh","release","create"])
        self.assertIn("--draft",commands[1])
        self.assertIn("--verify-tag",commands[1])
        self.assertEqual(commands[2][:3],["gh","release","edit"])
        self.assertIn("--draft=false",commands[2])

    def test_draft_digest_mismatch_never_publishes(self):
        draft = self.draft()
        draft["assets"][0]["digest"] = "sha256:"+"0"*64
        with patch.object(publisher,"verify_identity",return_value=NAME), \
                patch.object(publisher,"tag_sha",side_effect=[None,SHA]), \
                patch.object(publisher,"find_release",side_effect=[None,draft]), \
                patch.object(publisher.subprocess,"run") as run:
            with self.assertRaisesRegex(RuntimeError,"digest differs"):
                publisher.publish(SHA,self.assets)
        self.assertFalse(any(call.args[0][:3]==["gh","release","edit"] for call in run.call_args_list))

    def test_incomplete_draft_is_never_replaced(self):
        draft = self.draft()
        with patch.object(publisher,"verify_identity",return_value=NAME), \
                patch.object(publisher,"tag_sha",return_value=SHA), \
                patch.object(publisher,"find_release",return_value=draft), \
                patch.object(publisher.subprocess,"run") as run:
            with self.assertRaisesRegex(RuntimeError,"incomplete"):
                publisher.publish(SHA,self.assets)
            run.assert_not_called()

    def test_source_artifacts_reject_historical_and_unsafe_members(self):
        for path in ("../outside.py","pkg/legacy/old.py","pkg/model.pt","pkg/gpt.ipynb","pkg/demo.mp4"):
            with self.subTest(path=path),self.assertRaises(RuntimeError):
                build_python.check_member(path)
        build_python.check_member("pkg/src/gpt_scratch_lab/model.py")

    def test_smoke_requires_real_gui_and_optimizer_evidence(self):
        value=report(True)
        self.assertTrue(common.smoke_valid(value,frozen=True))
        value["training_steps"]=0
        self.assertFalse(common.smoke_valid(value,frozen=True))
        value=report(True)
        value["gui"]["widgets_constructed"]=False
        self.assertFalse(common.smoke_valid(value,frozen=True))


if __name__ == "__main__":
    unittest.main()
