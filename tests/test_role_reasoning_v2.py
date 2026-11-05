"""Integrity checks for the structured role-reasoning study."""
import unittest

from forge_core.neural.role_reasoning_v2 import (
    FINAL_OBJECTS, FINAL_PEOPLE, SKILLS, TRAIN_OBJECTS, TRAIN_PEOPLE,
    make_final, make_train, shuffle_targets, validate_split,
)


class TestRoleReasoningV2(unittest.TestCase):
    def test_training_datasets_are_deterministic_but_seed_independent(self):
        first = make_train(101, 400)
        same = make_train(101, 400)
        other = make_train(202, 400)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertFalse({row.prompt("name") for row in first} &
                         {row.prompt("name") for row in other})

    def test_episode_pairs_reverse_roles_and_all_rows_have_gold_evidence(self):
        cases = make_train(101, 160)
        episodes = {}
        for row in cases:
            episodes.setdefault(row.episode, []).append(row)
        self.assertEqual(len(episodes), 40)
        for rows in episodes.values():
            self.assertEqual(len(rows), 4)
            self.assertEqual({row.skill for row in rows}, set(SKILLS))
            giver_rows = [row for row in rows if row.skill == "giver"]
            recipient_rows = [row for row in rows if row.skill == "recipient"]
            self.assertEqual({row.giver for row in rows}, {row.recipient for row in rows})
            self.assertEqual({row.name_answer for row in giver_rows},
                             {row.giver for row in giver_rows})
            self.assertEqual({row.name_answer for row in recipient_rows},
                             {row.recipient for row in recipient_rows})
            self.assertNotEqual(giver_rows[0].giver, giver_rows[1].giver)
            self.assertNotEqual(recipient_rows[0].recipient,
                                recipient_rows[1].recipient)
            for row in rows:
                self.assertIn(row.giver, row.story)
                self.assertIn(row.recipient, row.story)
                self.assertIn(row.item, row.story)
                self.assertIn(row.letter_answer, {"A", "B"})
                self.assertIn(row.name_answer, {row.option_a, row.option_b})

    def test_letter_mapping_is_balanced_and_matches_name_answer(self):
        cases = make_train(314, 800)
        for skill in SKILLS:
            rows = [row for row in cases if row.skill == skill]
            self.assertEqual(sum(row.letter_answer == "A" for row in rows),
                             len(rows) // 2)
            for row in rows:
                expected = row.option_a if row.letter_answer == "A" else row.option_b
                self.assertEqual(expected, row.name_answer)

    def test_final_split_is_disjoint_and_includes_unseen_verbs(self):
        self.assertFalse(set(TRAIN_PEOPLE) & set(FINAL_PEOPLE))
        self.assertFalse(set(TRAIN_OBJECTS) & set(FINAL_OBJECTS))
        train = make_train(502, 800)
        final = make_final(82001, 60)
        self.assertEqual(final, make_final(82001, 60))
        validate_split(train, final)
        self.assertEqual(len(final), 240)
        self.assertEqual(sum(row.stratum == "unseen_verb" for row in final), 120)
        self.assertTrue(all("entrusted" in row.story or "delivered" in row.story
                            for row in final if row.stratum == "unseen_verb"))

    def test_shuffled_control_keeps_prompts_and_skill_target_counts(self):
        cases = make_train(88, 160)
        shuffled = shuffle_targets(cases, 77)
        self.assertEqual(len(shuffled), len(cases))
        self.assertNotEqual(shuffled, [row.target("name") for row in cases])
        for skill in SKILLS:
            gold = sorted(row.target("name") for row in cases if row.skill == skill)
            null = sorted(shuffled[i] for i, row in enumerate(cases)
                          if row.skill == skill)
            self.assertEqual(gold, null)

    def test_bounds(self):
        with self.assertRaises(ValueError):
            make_train(1, 42)
        with self.assertRaises(ValueError):
            make_final(episodes=19)


if __name__ == "__main__":
    unittest.main()
