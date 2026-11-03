"""Forge Teacher v1: verify lessons, prevent label leaks and resist bad advice."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
from subprocess import CompletedProcess

from forge_core.learning.teacher_v1 import (
    RULES, SKILL_SET, audit_bundle, make_plan, plan_quotas,
    validate_development_scores, validate_teacher_proposal, write_bundle,
)
from forge_core.language.local_teacher import LocalTeacher


class TestForgeTeacherV1(unittest.TestCase):
    def test_verified_lessons_are_reproducible_and_disjoint(self):
        plan, train, dev = make_plan(seed=1701,train_count=240,dev_count=120)
        other = make_plan(seed=1701,train_count=240,dev_count=120)
        self.assertEqual((plan,train,dev),other)
        self.assertEqual(len(train),240)
        self.assertEqual(len(dev),120)
        self.assertEqual(sum(plan["quotas"].values()),240)
        self.assertEqual({x.skill for x in train},SKILL_SET)
        self.assertFalse({x.prompt for x in train} & {x.prompt for x in dev})
        self.assertFalse(set(x.id for x in train) & set(x.id for x in dev))
        for row in train+dev:
            self.assertEqual(row.verified_rule,RULES[row.skill])
            self.assertIn(row.answer.casefold(),row.prompt.casefold())
            self.assertTrue(row.prompt.endswith("\nAnswer:"))

    def test_adaptive_focus_changes_counts_not_answers(self):
        score={name:{"correct":8,"total":10} for name in SKILL_SET}
        score["copy"]={"correct":0,"total":10}
        advice={"focus_skill":"recipient","reason":"Names have not transferred."}
        plan, train, _ = make_plan(progress=score,teacher_proposal=advice)
        self.assertGreater(plan["quotas"]["copy"],plan["quotas"]["color"])
        self.assertGreater(plan["quotas"]["recipient"],plan["quotas"]["color"])
        self.assertEqual(len(train),240)
        self.assertFalse(plan["teacher_can_change_gold_answers"])
        for invalid in [
            {"focus_skill":"private_data","reason":"No"},
            {"focus_skill":"copy","reason":"x","answer":"wrong"},
            {"focus_skill":"copy","reason":"\n"},
        ]:
            with self.assertRaises(ValueError):
                validate_teacher_proposal(invalid)

    def test_reject_raw_messages_in_development_scores(self):
        good={name:{"correct":3,"total":7} for name in SKILL_SET}
        self.assertEqual(validate_development_scores(good),good)
        with self.assertRaisesRegex(ValueError,"exactly six"):
            validate_development_scores({"email":"private"})
        good["copy"]["private_observation"]="private"
        with self.assertRaisesRegex(ValueError,"Only aggregate"):
            validate_development_scores(good)

    def test_bundle_integrity_and_no_gold_relabeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            manifest=write_bundle(folder)
            train, dev=audit_bundle(folder)
            self.assertEqual((len(train),len(dev)),(240,120))
            self.assertEqual(len(manifest["files_sha256"]),4)
            self.assertNotIn('"answer"', (folder/"development-questions.jsonl").read_text())
            original=(folder/"train.jsonl").read_bytes()
            # Invalid even if attacker updates a naive checksum:
            modified=original.replace(b'"answer":"', b'"answer":"incorrect_',1)
            (folder/"train.jsonl").write_bytes(modified)
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                audit_bundle(folder)
            (folder/"train.jsonl").write_bytes(original)
            audit_bundle(folder)

    def test_local_ai_teacher_gets_only_counts_and_is_advisory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            binary=root/"llama-cli";model=root/"teacher.gguf"
            binary.write_text("stub");model.write_bytes(b"stub")
            runner=Mock(return_value=CompletedProcess(
                ["llama-cli"],0,
                '{"focus_skill":"giver","reason":"Role questions need work."}',
                ""))
            teacher=LocalTeacher(binary,model,run=runner)
            scores={k:{"correct":1,"total":10} for k in SKILL_SET}
            suggestion=teacher.curriculum_proposal(scores)
            self.assertEqual(suggestion["focus_skill"],"giver")
            argv=runner.call_args.args[0]
            self.assertEqual(argv[0],str(binary))
            self.assertIn("aggregate development", " ".join(argv).lower())
            bad=Mock(return_value=CompletedProcess(["llama-cli"],0,
                '{"focus_skill":"giver","reason":"Fine","answer":"fake"}',""))
            teacher=LocalTeacher(binary,model,run=bad)
            with self.assertRaises(ValueError):
                teacher.curriculum_proposal(scores)


if __name__=="__main__":
    unittest.main()
