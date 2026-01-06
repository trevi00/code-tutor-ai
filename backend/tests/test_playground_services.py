"""Tests for Playground Application Services."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from code_tutor.playground.domain.entities import (
    Playground,
    CodeTemplate,
)
from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)
from code_tutor.playground.application.services import (
    PlaygroundService,
    TemplateService,
)
from code_tutor.playground.application.dto import (
    CreatePlaygroundRequest,
    UpdatePlaygroundRequest,
    ForkPlaygroundRequest,
)
from code_tutor.shared.exceptions import NotFoundError, ForbiddenError


@pytest.fixture
def mock_playground_repo():
    """Create mock playground repository."""
    return AsyncMock()


@pytest.fixture
def mock_history_repo():
    """Create mock execution history repository."""
    return AsyncMock()


@pytest.fixture
def mock_template_repo():
    """Create mock template repository."""
    return AsyncMock()


@pytest.fixture
def playground_service(mock_playground_repo, mock_history_repo):
    """Create PlaygroundService with mocks."""
    with patch("code_tutor.playground.application.services.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(
            SANDBOX_MEMORY_LIMIT_MB=128,
            SANDBOX_CPU_LIMIT=0.5,
        )
        service = PlaygroundService(
            playground_repo=mock_playground_repo,
            history_repo=mock_history_repo,
            use_docker=False,  # Use mock sandbox
        )
        return service


@pytest.fixture
def template_service(mock_template_repo):
    """Create TemplateService with mocks."""
    return TemplateService(template_repo=mock_template_repo)


@pytest.fixture
def sample_playground():
    """Create sample playground."""
    return Playground.create(
        owner_id=uuid4(),
        title="Test Playground",
        code="print('hello')",
        language=PlaygroundLanguage.PYTHON,
        description="Test description",
        visibility=PlaygroundVisibility.PUBLIC,
    )


@pytest.fixture
def sample_template():
    """Create sample code template."""
    return CodeTemplate(
        id=uuid4(),
        title="Two Pointers",
        description="Two pointers pattern",
        code="def two_pointers(arr): pass",
        language=PlaygroundLanguage.PYTHON,
        category=TemplateCategory.ALGORITHM,
        tags=["algorithm", "array"],
        usage_count=10,
    )


class TestPlaygroundServiceCreate:
    """Tests for create operations."""

    @pytest.mark.asyncio
    async def test_create_playground(self, playground_service, mock_playground_repo):
        """Test creating a new playground."""
        user_id = uuid4()
        request = CreatePlaygroundRequest(
            title="My Playground",
            description="Description",
            code="print('test')",
            language="python",
            visibility="public",
            stdin="input",
        )

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        result = await playground_service.create_playground(request, user_id)

        assert result.title == "My Playground"
        assert result.description == "Description"
        assert result.code == "print('test')"
        assert result.language == "python"
        assert result.visibility == "public"
        mock_playground_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_playground_with_default_code(
        self, playground_service, mock_playground_repo
    ):
        """Test creating playground with default code."""
        user_id = uuid4()
        request = CreatePlaygroundRequest(
            title="My Playground",
            language="python",
            code="",  # Empty code
        )

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        result = await playground_service.create_playground(request, user_id)

        assert result.title == "My Playground"
        # Should use default Python code
        assert "def main()" in result.code or "print" in result.code


class TestPlaygroundServiceGet:
    """Tests for get operations."""

    @pytest.mark.asyncio
    async def test_get_playground_found(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test getting an existing playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        result = await playground_service.get_playground(
            sample_playground.id, sample_playground.owner_id
        )

        assert result.id == sample_playground.id
        assert result.title == sample_playground.title

    @pytest.mark.asyncio
    async def test_get_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test getting non-existent playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.get_playground(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_get_playground_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test getting private playground by non-owner."""
        owner_id = uuid4()
        other_id = uuid4()
        private_playground = Playground.create(
            owner_id=owner_id,
            title="Private",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        mock_playground_repo.get_by_id = AsyncMock(return_value=private_playground)

        with pytest.raises(ForbiddenError):
            await playground_service.get_playground(private_playground.id, other_id)

    @pytest.mark.asyncio
    async def test_get_playground_by_share_code(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test getting playground by share code."""
        mock_playground_repo.get_by_share_code = AsyncMock(return_value=sample_playground)

        result = await playground_service.get_playground_by_share_code(
            sample_playground.share_code
        )

        assert result.id == sample_playground.id

    @pytest.mark.asyncio
    async def test_get_playground_by_share_code_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test getting by invalid share code."""
        mock_playground_repo.get_by_share_code = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.get_playground_by_share_code("invalid")


class TestPlaygroundServiceUpdate:
    """Tests for update operations."""

    @pytest.mark.asyncio
    async def test_update_playground(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test updating a playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        request = UpdatePlaygroundRequest(
            title="Updated Title",
            code="print('updated')",
        )

        result = await playground_service.update_playground(
            sample_playground.id, request, sample_playground.owner_id
        )

        assert result.title == "Updated Title"
        assert result.code == "print('updated')"

    @pytest.mark.asyncio
    async def test_update_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test updating non-existent playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.update_playground(
                uuid4(),
                UpdatePlaygroundRequest(title="New"),
                uuid4(),
            )

    @pytest.mark.asyncio
    async def test_update_playground_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test updating playground by non-owner."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        with pytest.raises(ForbiddenError):
            await playground_service.update_playground(
                sample_playground.id,
                UpdatePlaygroundRequest(title="New"),
                uuid4(),  # Different user
            )


class TestPlaygroundServiceDelete:
    """Tests for delete operations."""

    @pytest.mark.asyncio
    async def test_delete_playground(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test deleting a playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)
        mock_playground_repo.delete = AsyncMock()

        await playground_service.delete_playground(
            sample_playground.id, sample_playground.owner_id
        )

        mock_playground_repo.delete.assert_called_once_with(sample_playground.id)

    @pytest.mark.asyncio
    async def test_delete_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test deleting non-existent playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.delete_playground(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_delete_playground_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test deleting playground by non-owner."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        with pytest.raises(ForbiddenError):
            await playground_service.delete_playground(
                sample_playground.id, uuid4()
            )


class TestPlaygroundServiceFork:
    """Tests for fork operations."""

    @pytest.mark.asyncio
    async def test_fork_playground(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test forking a playground."""
        new_owner_id = uuid4()
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        result = await playground_service.fork_playground(
            sample_playground.id, new_owner_id
        )

        assert result.owner_id == new_owner_id
        assert result.is_forked is True
        assert result.forked_from_id == sample_playground.id
        assert "(Fork)" in result.title
        # save called twice: original (update fork_count) + forked
        assert mock_playground_repo.save.call_count == 2

    @pytest.mark.asyncio
    async def test_fork_playground_with_title(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test forking with custom title."""
        new_owner_id = uuid4()
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        result = await playground_service.fork_playground(
            sample_playground.id,
            new_owner_id,
            ForkPlaygroundRequest(title="My Custom Fork"),
        )

        assert result.title == "My Custom Fork"

    @pytest.mark.asyncio
    async def test_fork_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test forking non-existent playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.fork_playground(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_fork_private_playground_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test forking private playground by non-owner."""
        owner_id = uuid4()
        private_playground = Playground.create(
            owner_id=owner_id,
            title="Private",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        mock_playground_repo.get_by_id = AsyncMock(return_value=private_playground)

        with pytest.raises(ForbiddenError):
            await playground_service.fork_playground(
                private_playground.id, uuid4()
            )


class TestPlaygroundServiceList:
    """Tests for list operations."""

    @pytest.mark.asyncio
    async def test_get_user_playgrounds(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test getting user's playgrounds."""
        user_id = sample_playground.owner_id
        mock_playground_repo.get_user_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await playground_service.get_user_playgrounds(user_id)

        assert result.total == 1
        assert len(result.playgrounds) == 1
        assert result.playgrounds[0].id == sample_playground.id

    @pytest.mark.asyncio
    async def test_get_public_playgrounds(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test getting public playgrounds."""
        mock_playground_repo.get_public_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await playground_service.get_public_playgrounds()

        assert result.total == 1
        assert len(result.playgrounds) == 1

    @pytest.mark.asyncio
    async def test_get_public_playgrounds_with_language_filter(
        self, playground_service, mock_playground_repo
    ):
        """Test getting public playgrounds with language filter."""
        mock_playground_repo.get_public_playgrounds = AsyncMock(return_value=[])

        await playground_service.get_public_playgrounds(language="python")

        mock_playground_repo.get_public_playgrounds.assert_called_once_with(
            PlaygroundLanguage.PYTHON, 20, 0
        )

    @pytest.mark.asyncio
    async def test_get_popular_playgrounds(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test getting popular playgrounds."""
        mock_playground_repo.get_popular_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await playground_service.get_popular_playgrounds(limit=10)

        assert result.total == 1
        mock_playground_repo.get_popular_playgrounds.assert_called_once_with(10)

    @pytest.mark.asyncio
    async def test_search_playgrounds(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test searching playgrounds."""
        mock_playground_repo.search_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await playground_service.search_playgrounds("test")

        assert result.total == 1
        mock_playground_repo.search_playgrounds.assert_called_once()


class TestPlaygroundServiceShareCode:
    """Tests for share code operations."""

    @pytest.mark.asyncio
    async def test_regenerate_share_code(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test regenerating share code."""
        original_code = sample_playground.share_code
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        new_code = await playground_service.regenerate_share_code(
            sample_playground.id, sample_playground.owner_id
        )

        assert new_code != original_code
        mock_playground_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_regenerate_share_code_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test regenerating share code for non-existent playground."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await playground_service.regenerate_share_code(uuid4(), uuid4())

    @pytest.mark.asyncio
    async def test_regenerate_share_code_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test regenerating share code by non-owner."""
        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        with pytest.raises(ForbiddenError):
            await playground_service.regenerate_share_code(
                sample_playground.id, uuid4()
            )


class TestPlaygroundServiceLanguages:
    """Tests for language operations."""

    def test_get_supported_languages(self, playground_service):
        """Test getting supported languages."""
        result = playground_service.get_supported_languages()

        assert len(result.languages) > 0
        language_ids = [lang.id for lang in result.languages]
        assert "python" in language_ids
        assert "javascript" in language_ids

    def test_get_default_code(self, playground_service):
        """Test getting default code for language."""
        code = playground_service.get_default_code("python")

        assert len(code) > 0
        assert "def main()" in code or "print" in code


class TestTemplateService:
    """Tests for TemplateService."""

    @pytest.mark.asyncio
    async def test_get_templates(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test getting templates."""
        mock_template_repo.get_all = AsyncMock(return_value=[sample_template])

        result = await template_service.get_templates()

        assert result.total == 1
        assert len(result.templates) == 1
        assert result.templates[0].id == sample_template.id

    @pytest.mark.asyncio
    async def test_get_templates_with_filters(
        self, template_service, mock_template_repo
    ):
        """Test getting templates with filters."""
        mock_template_repo.get_all = AsyncMock(return_value=[])

        await template_service.get_templates(
            category="algorithm",
            language="python",
        )

        mock_template_repo.get_all.assert_called_once_with(
            TemplateCategory.ALGORITHM,
            PlaygroundLanguage.PYTHON,
        )

    @pytest.mark.asyncio
    async def test_get_template(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test getting a single template."""
        mock_template_repo.get_by_id = AsyncMock(return_value=sample_template)

        async def save_template(template):
            return template

        mock_template_repo.save = AsyncMock(side_effect=save_template)

        result = await template_service.get_template(sample_template.id)

        assert result.id == sample_template.id
        # Should increment usage
        mock_template_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_template_not_found(
        self, template_service, mock_template_repo
    ):
        """Test getting non-existent template."""
        mock_template_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await template_service.get_template(uuid4())

    @pytest.mark.asyncio
    async def test_get_popular_templates(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test getting popular templates."""
        mock_template_repo.get_popular = AsyncMock(return_value=[sample_template])

        result = await template_service.get_popular_templates(limit=10)

        assert result.total == 1
        mock_template_repo.get_popular.assert_called_once_with(10)
