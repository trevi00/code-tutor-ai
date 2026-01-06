"""Tests for Playground Domain Entities and Value Objects."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from code_tutor.playground.domain.entities import (
    Playground,
    CodeTemplate,
    ExecutionHistory,
    generate_share_code,
    utc_now,
)
from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
    LANGUAGE_CONFIG,
    DEFAULT_CODE,
)


class TestPlaygroundLanguage:
    """Tests for PlaygroundLanguage enum."""

    def test_language_values(self):
        """Test all language values exist."""
        assert PlaygroundLanguage.PYTHON.value == "python"
        assert PlaygroundLanguage.JAVASCRIPT.value == "javascript"
        assert PlaygroundLanguage.TYPESCRIPT.value == "typescript"
        assert PlaygroundLanguage.JAVA.value == "java"
        assert PlaygroundLanguage.CPP.value == "cpp"
        assert PlaygroundLanguage.C.value == "c"
        assert PlaygroundLanguage.GO.value == "go"
        assert PlaygroundLanguage.RUST.value == "rust"

    def test_language_from_string(self):
        """Test creating language from string."""
        assert PlaygroundLanguage("python") == PlaygroundLanguage.PYTHON
        assert PlaygroundLanguage("javascript") == PlaygroundLanguage.JAVASCRIPT

    def test_all_languages_have_config(self):
        """Test all languages have configuration."""
        for lang in PlaygroundLanguage:
            assert lang in LANGUAGE_CONFIG
            config = LANGUAGE_CONFIG[lang]
            assert "extension" in config
            assert "docker_image" in config
            assert "run_command" in config
            assert "display_name" in config

    def test_all_languages_have_default_code(self):
        """Test all languages have default code."""
        for lang in PlaygroundLanguage:
            assert lang in DEFAULT_CODE
            assert len(DEFAULT_CODE[lang]) > 0


class TestPlaygroundVisibility:
    """Tests for PlaygroundVisibility enum."""

    def test_visibility_values(self):
        """Test all visibility values exist."""
        assert PlaygroundVisibility.PRIVATE.value == "private"
        assert PlaygroundVisibility.UNLISTED.value == "unlisted"
        assert PlaygroundVisibility.PUBLIC.value == "public"

    def test_visibility_from_string(self):
        """Test creating visibility from string."""
        assert PlaygroundVisibility("private") == PlaygroundVisibility.PRIVATE
        assert PlaygroundVisibility("unlisted") == PlaygroundVisibility.UNLISTED
        assert PlaygroundVisibility("public") == PlaygroundVisibility.PUBLIC


class TestTemplateCategory:
    """Tests for TemplateCategory enum."""

    def test_category_values(self):
        """Test all category values exist."""
        assert TemplateCategory.BASIC.value == "basic"
        assert TemplateCategory.ALGORITHM.value == "algorithm"
        assert TemplateCategory.DATA_STRUCTURE.value == "data_structure"
        assert TemplateCategory.PATTERN.value == "pattern"
        assert TemplateCategory.SNIPPET.value == "snippet"
        assert TemplateCategory.STARTER.value == "starter"
        assert TemplateCategory.UTILITY.value == "utility"


class TestPlayground:
    """Tests for Playground entity."""

    def test_playground_creation(self):
        """Test creating a playground directly."""
        playground_id = uuid4()
        owner_id = uuid4()
        playground = Playground(
            id=playground_id,
            owner_id=owner_id,
            title="My Playground",
            description="Test description",
            code="print('hello')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        assert playground.id == playground_id
        assert playground.owner_id == owner_id
        assert playground.title == "My Playground"
        assert playground.language == PlaygroundLanguage.PYTHON
        assert playground.visibility == PlaygroundVisibility.PRIVATE

    def test_playground_create_factory(self):
        """Test playground factory method."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test Playground",
            code="print('test')",
            language=PlaygroundLanguage.PYTHON,
            description="Description",
            visibility=PlaygroundVisibility.PUBLIC,
            stdin="input data",
        )
        assert playground.id is not None
        assert playground.owner_id == owner_id
        assert playground.title == "Test Playground"
        assert playground.code == "print('test')"
        assert playground.language == PlaygroundLanguage.PYTHON
        assert playground.visibility == PlaygroundVisibility.PUBLIC
        assert playground.stdin == "input data"
        assert playground.is_forked is False
        assert playground.run_count == 0
        assert playground.fork_count == 0
        assert playground.share_code is not None

    def test_playground_default_values(self):
        """Test playground default values."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )
        assert playground.description == ""
        assert playground.visibility == PlaygroundVisibility.PRIVATE
        assert playground.stdin == ""

    def test_playground_fork(self):
        """Test forking a playground."""
        owner_id = uuid4()
        new_owner_id = uuid4()
        original = Playground.create(
            owner_id=owner_id,
            title="Original",
            code="print('original')",
            language=PlaygroundLanguage.PYTHON,
            description="Original description",
            visibility=PlaygroundVisibility.PUBLIC,
            stdin="input",
        )
        original_fork_count = original.fork_count

        forked = original.fork(new_owner_id)

        # Original should have incremented fork count
        assert original.fork_count == original_fork_count + 1

        # Forked should have new properties
        assert forked.id != original.id
        assert forked.owner_id == new_owner_id
        assert forked.title == "Original (Fork)"
        assert forked.code == original.code
        assert forked.language == original.language
        assert forked.description == original.description
        assert forked.stdin == original.stdin
        assert forked.visibility == PlaygroundVisibility.PRIVATE  # Forks are private
        assert forked.is_forked is True
        assert forked.forked_from_id == original.id
        assert forked.run_count == 0
        assert forked.fork_count == 0

    def test_playground_update(self):
        """Test updating a playground."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Original",
            code="original code",
            language=PlaygroundLanguage.PYTHON,
        )

        playground.update(
            title="Updated Title",
            description="New description",
            code="updated code",
            language=PlaygroundLanguage.JAVASCRIPT,
            visibility=PlaygroundVisibility.PUBLIC,
            stdin="new input",
        )

        assert playground.title == "Updated Title"
        assert playground.description == "New description"
        assert playground.code == "updated code"
        assert playground.language == PlaygroundLanguage.JAVASCRIPT
        assert playground.visibility == PlaygroundVisibility.PUBLIC
        assert playground.stdin == "new input"

    def test_playground_update_partial(self):
        """Test partial update of playground."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Original",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            description="Original description",
        )

        playground.update(title="New Title")

        assert playground.title == "New Title"
        assert playground.code == "code"
        assert playground.description == "Original description"

    def test_playground_increment_run_count(self):
        """Test incrementing run count."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )
        assert playground.run_count == 0

        playground.increment_run_count()
        assert playground.run_count == 1

        playground.increment_run_count()
        assert playground.run_count == 2

    def test_playground_regenerate_share_code(self):
        """Test regenerating share code."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )
        original_code = playground.share_code

        new_code = playground.regenerate_share_code()

        assert new_code != original_code
        assert playground.share_code == new_code

    def test_playground_is_owner(self):
        """Test is_owner check."""
        owner_id = uuid4()
        other_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )

        assert playground.is_owner(owner_id) is True
        assert playground.is_owner(other_id) is False

    def test_playground_can_view_public(self):
        """Test can_view for public playground."""
        owner_id = uuid4()
        other_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )

        assert playground.can_view(owner_id) is True
        assert playground.can_view(other_id) is True
        assert playground.can_view(None) is True

    def test_playground_can_view_unlisted(self):
        """Test can_view for unlisted playground."""
        owner_id = uuid4()
        other_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.UNLISTED,
        )

        # Anyone with link can view unlisted
        assert playground.can_view(owner_id) is True
        assert playground.can_view(other_id) is True
        assert playground.can_view(None) is True

    def test_playground_can_view_private(self):
        """Test can_view for private playground."""
        owner_id = uuid4()
        other_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )

        assert playground.can_view(owner_id) is True
        assert playground.can_view(other_id) is False
        assert playground.can_view(None) is False

    def test_playground_can_edit(self):
        """Test can_edit check."""
        owner_id = uuid4()
        other_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )

        assert playground.can_edit(owner_id) is True
        assert playground.can_edit(other_id) is False
        assert playground.can_edit(None) is False

    def test_playground_to_dict(self):
        """Test to_dict method."""
        owner_id = uuid4()
        playground = Playground.create(
            owner_id=owner_id,
            title="Test Playground",
            code="print('hello')",
            language=PlaygroundLanguage.PYTHON,
            description="Test description",
            visibility=PlaygroundVisibility.PUBLIC,
        )

        data = playground.to_dict()

        assert data["id"] == str(playground.id)
        assert data["owner_id"] == str(owner_id)
        assert data["title"] == "Test Playground"
        assert data["code"] == "print('hello')"
        assert data["language"] == "python"
        assert data["visibility"] == "public"
        assert data["description"] == "Test description"
        assert "share_code" in data
        assert "created_at" in data
        assert "updated_at" in data


class TestCodeTemplate:
    """Tests for CodeTemplate entity."""

    def test_template_creation(self):
        """Test creating a code template."""
        template_id = uuid4()
        template = CodeTemplate(
            id=template_id,
            title="Two Pointers",
            description="Two pointers pattern",
            code="def two_pointers(arr): pass",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.ALGORITHM,
            tags=["algorithm", "array"],
        )
        assert template.id == template_id
        assert template.title == "Two Pointers"
        assert template.category == TemplateCategory.ALGORITHM
        assert template.tags == ["algorithm", "array"]
        assert template.usage_count == 0

    def test_template_default_values(self):
        """Test template default values."""
        template = CodeTemplate(
            id=uuid4(),
            title="Test",
            description="Description",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.SNIPPET,
        )
        assert template.tags == []
        assert template.usage_count == 0

    def test_template_increment_usage(self):
        """Test incrementing usage count."""
        template = CodeTemplate(
            id=uuid4(),
            title="Test",
            description="Description",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.SNIPPET,
        )
        assert template.usage_count == 0

        template.increment_usage()
        assert template.usage_count == 1

        template.increment_usage()
        assert template.usage_count == 2

    def test_template_to_dict(self):
        """Test to_dict method."""
        template_id = uuid4()
        template = CodeTemplate(
            id=template_id,
            title="Test Template",
            description="Description",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.ALGORITHM,
            tags=["tag1", "tag2"],
            usage_count=10,
        )

        data = template.to_dict()

        assert data["id"] == str(template_id)
        assert data["title"] == "Test Template"
        assert data["description"] == "Description"
        assert data["code"] == "code"
        assert data["language"] == "python"
        assert data["category"] == "algorithm"
        assert data["tags"] == ["tag1", "tag2"]
        assert data["usage_count"] == 10


class TestExecutionHistory:
    """Tests for ExecutionHistory entity."""

    def test_execution_history_creation(self):
        """Test creating execution history."""
        history_id = uuid4()
        playground_id = uuid4()
        user_id = uuid4()
        history = ExecutionHistory(
            id=history_id,
            playground_id=playground_id,
            user_id=user_id,
            code="print('hello')",
            stdin="",
            stdout="hello\n",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            is_success=True,
        )
        assert history.id == history_id
        assert history.playground_id == playground_id
        assert history.user_id == user_id
        assert history.is_success is True
        assert history.exit_code == 0

    def test_execution_history_no_user(self):
        """Test execution history without user (anonymous)."""
        history = ExecutionHistory(
            id=uuid4(),
            playground_id=uuid4(),
            user_id=None,
            code="code",
            stdin="",
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_ms=10.0,
            is_success=True,
        )
        assert history.user_id is None

    def test_execution_history_to_dict(self):
        """Test to_dict method."""
        history_id = uuid4()
        playground_id = uuid4()
        user_id = uuid4()
        history = ExecutionHistory(
            id=history_id,
            playground_id=playground_id,
            user_id=user_id,
            code="print('hello')",
            stdin="input",
            stdout="hello\n",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            is_success=True,
        )

        data = history.to_dict()

        assert data["id"] == str(history_id)
        assert data["playground_id"] == str(playground_id)
        assert data["user_id"] == str(user_id)
        assert data["stdout"] == "hello\n"
        assert data["stderr"] == ""
        assert data["exit_code"] == 0
        assert data["execution_time_ms"] == 50.0
        assert data["is_success"] is True

    def test_execution_history_to_dict_no_user(self):
        """Test to_dict method with no user."""
        history = ExecutionHistory(
            id=uuid4(),
            playground_id=uuid4(),
            user_id=None,
            code="code",
            stdin="",
            stdout="",
            stderr="",
            exit_code=0,
            execution_time_ms=10.0,
            is_success=True,
        )

        data = history.to_dict()

        assert data["user_id"] is None


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_generate_share_code(self):
        """Test share code generation."""
        code1 = generate_share_code()
        code2 = generate_share_code()

        assert code1 != code2
        assert len(code1) > 0
        assert len(code2) > 0

    def test_utc_now_returns_aware_datetime(self):
        """Test that utc_now returns timezone-aware datetime."""
        now = utc_now()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_utc_now_is_recent(self):
        """Test that utc_now returns current time."""
        before = datetime.now(timezone.utc)
        now = utc_now()
        after = datetime.now(timezone.utc)
        assert before <= now <= after
