from __future__ import annotations

from pathlib import Path

from agent_advanced import AdvancedAgent
from agent_baseline import BaselineAgent
from config import LabConfig, load_config
from memory_store import UserProfileStore


def make_config(tmp_path: Path) -> LabConfig:
    """Build an isolated LabConfig for tests with a low compaction threshold."""

    cfg = load_config()
    cfg.state_dir = tmp_path / "state"
    cfg.state_dir.mkdir(parents=True, exist_ok=True)
    cfg.compact_threshold_tokens = 80
    cfg.compact_keep_messages = 2
    return cfg


def test_user_markdown_read_write_edit(tmp_path: Path) -> None:
    """Verify `User.md` can be created, read, updated, and edited."""

    store = UserProfileStore(tmp_path / "profiles")
    user_id = "dungct"

    assert store.read_text(user_id) == ""
    assert store.file_size(user_id) == 0

    initial_md = "# User Profile\n\n- **name**: DũngCT\n- **location**: Đà Nẵng\n"
    written_path = store.write_text(user_id, initial_md)
    assert written_path.exists()
    assert store.read_text(user_id) == initial_md
    assert store.file_size(user_id) > 0

    changed = store.edit_text(user_id, "Đà Nẵng", "Huế")
    assert changed is True
    assert "Huế" in store.read_text(user_id)
    assert "Đà Nẵng" not in store.read_text(user_id)

    not_changed = store.edit_text(user_id, "Không tồn tại", "Giá trị mới")
    assert not_changed is False


def test_compact_trigger(tmp_path: Path) -> None:
    """Verify long threads trigger compaction and populate summary while trimming messages."""

    cfg = make_config(tmp_path)
    agent = AdvancedAgent(config=cfg, force_offline=True)

    long_turns = [
        "Lượt 1: Chúng ta đang thảo luận về kiến trúc bộ nhớ dài hạn và ngắn hạn cho AI agent trong môi trường thực tế.",
        "Lượt 2: Khi hội thoại kéo dài nhiều lượt, việc giữ nguyên văn toàn bộ lịch sử sẽ làm chi phí prompt tăng bậc hai.",
        "Lượt 3: Vì vậy cơ chế compact memory cần tự động tóm tắt các lượt cũ và chỉ giữ lại một vài lượt gần nhất.",
        "Lượt 4: Đồng thời hồ sơ người dùng ổn định nên được tách riêng ra file User.md để không bị mất khi nén ngữ cảnh.",
    ]

    for turn in long_turns:
        agent.reply("user_compact", "thread-compact-1", turn)

    assert agent.compaction_count("thread-compact-1") > 0
    ctx = agent.compact_memory.context("thread-compact-1")
    assert str(ctx["summary"]).startswith("Tóm tắt lịch sử:")
    assert len(ctx["messages"]) <= cfg.compact_keep_messages


def test_cross_session_recall(tmp_path: Path) -> None:
    """Verify AdvancedAgent remembers facts across sessions while BaselineAgent does not."""

    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    intro_turns = [
        "Chào bạn, mình tên là DũngCT.",
        "Mình ở Huế và đang làm MLOps engineer.",
        "Đồ uống yêu thích là cà phê sữa đá.",
    ]

    for turn in intro_turns:
        baseline.reply("dungct", "session-1", turn)
        advanced.reply("dungct", "session-1", turn)

    question = "Sang phiên mới rồi, mình tên gì, hiện ở đâu và làm nghề gì?"
    base_res = baseline.reply("dungct", "session-2", question)["response"]
    adv_res = advanced.reply("dungct", "session-2", question)["response"]

    assert "DũngCT" not in base_res
    assert "MLOps engineer" not in base_res

    assert "DũngCT" in adv_res
    assert "Huế" in adv_res
    assert "MLOps engineer" in adv_res


def test_compact_reduces_prompt_load_on_long_thread(tmp_path: Path) -> None:
    """Compare cumulative prompt load of BaselineAgent vs AdvancedAgent on a long thread."""

    cfg = make_config(tmp_path)
    baseline = BaselineAgent(config=cfg, force_offline=True)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    for i in range(12):
        long_msg = (
            f"Bản tin kỹ thuật số {i + 1}: Hệ thống AI agent trong sản xuất cần kiểm soát chặt "
            f"chẽ độ trễ và lượng prompt tokens xử lý qua mỗi lượt hội thoại dài để tối ưu chi phí vận hành."
        )
        baseline.reply("stress_user", "long-thread", long_msg)
        advanced.reply("stress_user", "long-thread", long_msg)

    assert advanced.compaction_count("long-thread") > 0
    assert advanced.prompt_token_usage("long-thread") < baseline.prompt_token_usage("long-thread")


def test_conflict_handling_and_correction(tmp_path: Path) -> None:
    """Bonus test: verify conflict resolution replaces outdated location and profession on correction."""

    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    advanced.reply("dungct", "t1", "Chào bạn, mình tên là DũngCT. Mình ở Đà Nẵng và đang làm backend engineer.")
    advanced.reply("dungct", "t2", "À, mình đính chính: giờ mình đang ở Huế chứ không còn ở Đà Nẵng mỗi ngày nữa.")
    advanced.reply("dungct", "t3", "Mình không còn làm backend engineer nữa, giờ chuyển sang MLOps engineer.")

    ans = advanced.reply("dungct", "t4-recall", "Hiện tại mình làm nghề gì và đang ở đâu?")["response"]
    assert "MLOps engineer" in ans
    assert "Huế" in ans
    assert "backend engineer" not in ans
    assert "Đà Nẵng" not in ans


def test_confidence_threshold_and_noise_filtering(tmp_path: Path) -> None:
    """Bonus test: verify questions and low-confidence joke/travel distractors do not pollute `User.md`."""

    cfg = make_config(tmp_path)
    advanced = AdvancedAgent(config=cfg, force_offline=True)

    advanced.reply("dungct_stress", "s1", "Mình tên là DũngCT Stress, nơi ở hiện tại là Đà Nẵng, nghề MLOps engineer.")
    advanced.reply(
        "dungct_stress",
        "s1",
        "Có lúc mình đùa với đồng nghiệp rằng hay là chuyển sang product manager, nhưng đó chỉ là câu đùa. "
        "Hà Nội chỉ là nơi mình vừa bay ra họp hai ngày với đối tác chứ không phải nơi ở hiện tại.",
    )

    ans = advanced.reply(
        "dungct_stress",
        "s2-recall",
        "Nếu ai đó nhắc Huế, Hà Nội hay product manager, đâu mới là nghề nghiệp và nơi ở hiện tại của mình?",
    )["response"]
    assert "MLOps engineer" in ans
    assert "Đà Nẵng" in ans
    assert "product manager" not in ans
    assert "Hà Nội" not in ans


def test_memory_decay_limits_profile_bloat(tmp_path: Path) -> None:
    """Bonus test: verify memory decay bounds optional transient facts in `User.md`."""

    store = UserProfileStore(tmp_path / "profiles", max_optional_facts=3)
    user_id = "decay_user"

    store.upsert_fact(user_id, "name", "DũngCT")
    for idx in range(6):
        store.upsert_fact(user_id, f"temp_topic_{idx}", f"Chủ đề tạm thời {idx}")

    facts = store.facts(user_id)
    assert facts.get("name") == "DũngCT"
    optional_keys = [k for k in facts if k.startswith("temp_topic_")]
    assert len(optional_keys) <= 3
    assert "temp_topic_5" in optional_keys