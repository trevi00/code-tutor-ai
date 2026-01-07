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


# ============== Route Tests ==============


class TestPlaygroundRoutesUnit:
    """Unit tests for playground routes."""

    def test_router_has_expected_routes(self):
        """Test that router has all expected routes configured."""
        from code_tutor.playground.interface.routes import router

        route_paths = [r.path for r in router.routes]

        expected_paths = [
            "/playground",
            "/playground/mine",
            "/playground/public",
            "/playground/popular",
            "/playground/search",
            "/playground/languages",
            "/playground/default-code",
            "/playground/share/{share_code}",
            "/playground/{playground_id}",
            "/playground/{playground_id}/execute",
            "/playground/{playground_id}/fork",
            "/playground/{playground_id}/regenerate-share-code",
            "/playground/templates/list",
            "/playground/templates/popular",
            "/playground/templates/{template_id}",
        ]

        for path in expected_paths:
            assert path in route_paths, f"Missing route: {path}"

    def test_router_prefix(self):
        """Test router has correct prefix."""
        from code_tutor.playground.interface.routes import router
        assert router.prefix == "/playground"

    def test_router_tags(self):
        """Test router has correct tags."""
        from code_tutor.playground.interface.routes import router
        assert "Playground" in router.tags


class TestGetPlaygroundService:
    """Tests for get_playground_service dependency."""

    @pytest.mark.asyncio
    async def test_get_playground_service_returns_service(self):
        """Test that get_playground_service returns a PlaygroundService instance."""
        from code_tutor.playground.interface.routes import get_playground_service
        from code_tutor.playground.application.services import PlaygroundService

        mock_db = MagicMock()
        service = await get_playground_service(mock_db)

        assert isinstance(service, PlaygroundService)


class TestGetTemplateService:
    """Tests for get_template_service dependency."""

    @pytest.mark.asyncio
    async def test_get_template_service_returns_service(self):
        """Test that get_template_service returns a TemplateService instance."""
        from code_tutor.playground.interface.routes import get_template_service
        from code_tutor.playground.application.services import TemplateService

        mock_db = MagicMock()
        service = await get_template_service(mock_db)

        assert isinstance(service, TemplateService)


class TestCreatePlaygroundRoute:
    """Tests for create_playground route."""

    @pytest.mark.asyncio
    async def test_create_playground_success(
        self, playground_service, mock_playground_repo
    ):
        """Test create_playground route success."""
        from code_tutor.playground.interface.routes import create_playground
        from code_tutor.playground.application.dto import PlaygroundDetailResponse
        from code_tutor.identity.application.dto import UserResponse

        user_id = uuid4()
        request = CreatePlaygroundRequest(
            title="Test",
            code="print('hello')",
            language="python",
        )

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = user_id

        result = await create_playground(
            request=request,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundDetailResponse)
        assert result.title == "Test"


class TestListMyPlaygroundsRoute:
    """Tests for list_my_playgrounds route."""

    @pytest.mark.asyncio
    async def test_list_my_playgrounds_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test list_my_playgrounds route success."""
        from code_tutor.playground.interface.routes import list_my_playgrounds
        from code_tutor.playground.application.dto import PlaygroundListResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_user_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await list_my_playgrounds(
            limit=20,
            offset=0,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundListResponse)
        assert result.total == 1


class TestListPublicPlaygroundsRoute:
    """Tests for list_public_playgrounds route."""

    @pytest.mark.asyncio
    async def test_list_public_playgrounds_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test list_public_playgrounds route success."""
        from code_tutor.playground.interface.routes import list_public_playgrounds
        from code_tutor.playground.application.dto import PlaygroundListResponse

        mock_playground_repo.get_public_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await list_public_playgrounds(
            language=None,
            limit=20,
            offset=0,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundListResponse)
        assert result.total == 1


class TestListPopularPlaygroundsRoute:
    """Tests for list_popular_playgrounds route."""

    @pytest.mark.asyncio
    async def test_list_popular_playgrounds_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test list_popular_playgrounds route success."""
        from code_tutor.playground.interface.routes import list_popular_playgrounds
        from code_tutor.playground.application.dto import PlaygroundListResponse

        mock_playground_repo.get_popular_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await list_popular_playgrounds(
            limit=10,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundListResponse)
        assert result.total == 1


class TestSearchPlaygroundsRoute:
    """Tests for search_playgrounds route."""

    @pytest.mark.asyncio
    async def test_search_playgrounds_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test search_playgrounds route success."""
        from code_tutor.playground.interface.routes import search_playgrounds
        from code_tutor.playground.application.dto import PlaygroundListResponse

        mock_playground_repo.search_playgrounds = AsyncMock(
            return_value=[sample_playground]
        )

        result = await search_playgrounds(
            q="test",
            language=None,
            limit=20,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundListResponse)
        assert result.total == 1


class TestListLanguagesRoute:
    """Tests for list_languages route."""

    @pytest.mark.asyncio
    async def test_list_languages_success(self, playground_service):
        """Test list_languages route success."""
        from code_tutor.playground.interface.routes import list_languages
        from code_tutor.playground.application.dto import LanguagesResponse

        result = await list_languages(service=playground_service)

        assert isinstance(result, LanguagesResponse)
        assert len(result.languages) > 0


class TestGetDefaultCodeRoute:
    """Tests for get_default_code route."""

    @pytest.mark.asyncio
    async def test_get_default_code_success(self, playground_service):
        """Test get_default_code route success."""
        from code_tutor.playground.interface.routes import get_default_code

        result = await get_default_code(
            language="python",
            service=playground_service,
        )

        assert "language" in result
        assert "code" in result
        assert result["language"] == "python"


class TestGetPlaygroundByShareCodeRoute:
    """Tests for get_playground_by_share_code route."""

    @pytest.mark.asyncio
    async def test_get_playground_by_share_code_found(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test get_playground_by_share_code route when found."""
        from code_tutor.playground.interface.routes import get_playground_by_share_code
        from code_tutor.playground.application.dto import PlaygroundDetailResponse

        mock_playground_repo.get_by_share_code = AsyncMock(return_value=sample_playground)

        result = await get_playground_by_share_code(
            share_code="abc123",
            service=playground_service,
        )

        assert isinstance(result, PlaygroundDetailResponse)
        assert result.id == sample_playground.id

    @pytest.mark.asyncio
    async def test_get_playground_by_share_code_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test get_playground_by_share_code route when not found."""
        from code_tutor.playground.interface.routes import get_playground_by_share_code
        from fastapi import HTTPException

        mock_playground_repo.get_by_share_code = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_playground_by_share_code(
                share_code="invalid",
                service=playground_service,
            )

        assert exc_info.value.status_code == 404


class TestGetPlaygroundRoute:
    """Tests for get_playground route."""

    @pytest.mark.asyncio
    async def test_get_playground_found(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test get_playground route when found."""
        from code_tutor.playground.interface.routes import get_playground
        from code_tutor.playground.application.dto import PlaygroundDetailResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await get_playground(
            playground_id=sample_playground.id,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundDetailResponse)
        assert result.id == sample_playground.id

    @pytest.mark.asyncio
    async def test_get_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test get_playground route when not found."""
        from code_tutor.playground.interface.routes import get_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await get_playground(
                playground_id=uuid4(),
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_playground_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test get_playground route when forbidden."""
        from code_tutor.playground.interface.routes import get_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        owner_id = uuid4()
        private_playground = Playground.create(
            owner_id=owner_id,
            title="Private",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        mock_playground_repo.get_by_id = AsyncMock(return_value=private_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await get_playground(
                playground_id=private_playground.id,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403


class TestUpdatePlaygroundRoute:
    """Tests for update_playground route."""

    @pytest.mark.asyncio
    async def test_update_playground_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test update_playground route success."""
        from code_tutor.playground.interface.routes import update_playground
        from code_tutor.playground.application.dto import PlaygroundDetailResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        request = UpdatePlaygroundRequest(title="Updated")

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await update_playground(
            playground_id=sample_playground.id,
            request=request,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundDetailResponse)
        assert result.title == "Updated"

    @pytest.mark.asyncio
    async def test_update_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test update_playground route when not found."""
        from code_tutor.playground.interface.routes import update_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await update_playground(
                playground_id=uuid4(),
                request=UpdatePlaygroundRequest(title="New"),
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_update_playground_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test update_playground route when forbidden."""
        from code_tutor.playground.interface.routes import update_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await update_playground(
                playground_id=sample_playground.id,
                request=UpdatePlaygroundRequest(title="New"),
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403


class TestDeletePlaygroundRoute:
    """Tests for delete_playground route."""

    @pytest.mark.asyncio
    async def test_delete_playground_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test delete_playground route success."""
        from code_tutor.playground.interface.routes import delete_playground
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)
        mock_playground_repo.delete = AsyncMock()

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await delete_playground(
            playground_id=sample_playground.id,
            current_user=mock_user,
            service=playground_service,
        )

        assert result == {"status": "deleted"}

    @pytest.mark.asyncio
    async def test_delete_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test delete_playground route when not found."""
        from code_tutor.playground.interface.routes import delete_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await delete_playground(
                playground_id=uuid4(),
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_playground_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test delete_playground route when forbidden."""
        from code_tutor.playground.interface.routes import delete_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await delete_playground(
                playground_id=sample_playground.id,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403


class TestForkPlaygroundRoute:
    """Tests for fork_playground route."""

    @pytest.mark.asyncio
    async def test_fork_playground_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test fork_playground route success."""
        from code_tutor.playground.interface.routes import fork_playground
        from code_tutor.playground.application.dto import PlaygroundDetailResponse
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        result = await fork_playground(
            playground_id=sample_playground.id,
            request=None,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, PlaygroundDetailResponse)
        assert result.is_forked is True

    @pytest.mark.asyncio
    async def test_fork_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test fork_playground route when not found."""
        from code_tutor.playground.interface.routes import fork_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await fork_playground(
                playground_id=uuid4(),
                request=None,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404


class TestRegenerateShareCodeRoute:
    """Tests for regenerate_share_code route."""

    @pytest.mark.asyncio
    async def test_regenerate_share_code_success(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test regenerate_share_code route success."""
        from code_tutor.playground.interface.routes import regenerate_share_code
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await regenerate_share_code(
            playground_id=sample_playground.id,
            current_user=mock_user,
            service=playground_service,
        )

        assert "share_code" in result

    @pytest.mark.asyncio
    async def test_regenerate_share_code_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test regenerate_share_code route when not found."""
        from code_tutor.playground.interface.routes import regenerate_share_code
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await regenerate_share_code(
                playground_id=uuid4(),
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404


class TestTemplateRoutesUnit:
    """Tests for template routes."""

    @pytest.mark.asyncio
    async def test_list_templates_success(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test list_templates route success."""
        from code_tutor.playground.interface.routes import list_templates
        from code_tutor.playground.application.dto import TemplateListResponse

        mock_template_repo.get_all = AsyncMock(return_value=[sample_template])

        result = await list_templates(
            category=None,
            language=None,
            service=template_service,
        )

        assert isinstance(result, TemplateListResponse)
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_list_popular_templates_success(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test list_popular_templates route success."""
        from code_tutor.playground.interface.routes import list_popular_templates
        from code_tutor.playground.application.dto import TemplateListResponse

        mock_template_repo.get_popular = AsyncMock(return_value=[sample_template])

        result = await list_popular_templates(
            limit=10,
            service=template_service,
        )

        assert isinstance(result, TemplateListResponse)
        assert result.total == 1

    @pytest.mark.asyncio
    async def test_get_template_found(
        self, template_service, mock_template_repo, sample_template
    ):
        """Test get_template route when found."""
        from code_tutor.playground.interface.routes import get_template
        from code_tutor.playground.application.dto import TemplateResponse

        mock_template_repo.get_by_id = AsyncMock(return_value=sample_template)

        async def save_template(template):
            return template

        mock_template_repo.save = AsyncMock(side_effect=save_template)

        result = await get_template(
            template_id=sample_template.id,
            service=template_service,
        )

        assert isinstance(result, TemplateResponse)
        assert result.id == sample_template.id

    @pytest.mark.asyncio
    async def test_get_template_not_found(
        self, template_service, mock_template_repo
    ):
        """Test get_template route when not found."""
        from code_tutor.playground.interface.routes import get_template
        from fastapi import HTTPException

        mock_template_repo.get_by_id = AsyncMock(return_value=None)

        with pytest.raises(HTTPException) as exc_info:
            await get_template(
                template_id=uuid4(),
                service=template_service,
            )

        assert exc_info.value.status_code == 404


# ============== Execute Playground Tests ==============


class TestPlaygroundServiceExecute:
    """Tests for execute_playground operations."""

    @pytest.mark.asyncio
    async def test_execute_playground_success(
        self, playground_service, mock_playground_repo, mock_history_repo, sample_playground
    ):
        """Test executing playground code successfully."""
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest
        from code_tutor.execution.domain.value_objects import ExecutionStatus

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)
        mock_history_repo.save = AsyncMock()

        request = ExecutePlaygroundRequest(
            code="print('hello')",
            stdin="",
            timeout_seconds=10,
        )

        result = await playground_service.execute_playground(
            sample_playground.id, request, sample_playground.owner_id
        )

        assert result.is_success is True
        assert result.status == ExecutionStatus.SUCCESS.value
        mock_playground_repo.save.assert_called_once()  # Run count incremented
        mock_history_repo.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_playground_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test executing non-existent playground."""
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        request = ExecutePlaygroundRequest(code="print('hello')")

        with pytest.raises(NotFoundError):
            await playground_service.execute_playground(uuid4(), request, uuid4())

    @pytest.mark.asyncio
    async def test_execute_playground_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test executing private playground by non-owner."""
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest

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

        request = ExecutePlaygroundRequest(code="print('hello')")

        with pytest.raises(ForbiddenError):
            await playground_service.execute_playground(
                private_playground.id, request, other_id
            )

    @pytest.mark.asyncio
    async def test_execute_playground_uses_saved_code_when_none(
        self, playground_service, mock_playground_repo, mock_history_repo, sample_playground
    ):
        """Test executing uses saved code when request code is None."""
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)
        mock_history_repo.save = AsyncMock()

        request = ExecutePlaygroundRequest(code=None, stdin="")

        result = await playground_service.execute_playground(
            sample_playground.id, request, sample_playground.owner_id
        )

        # Should execute successfully using saved code
        assert result.is_success is True

    @pytest.mark.asyncio
    async def test_execute_playground_anonymous_user(
        self, playground_service, mock_playground_repo, mock_history_repo, sample_playground
    ):
        """Test executing public playground by anonymous user."""
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)
        mock_history_repo.save = AsyncMock()

        request = ExecutePlaygroundRequest(code="print('hello')")

        # Execute with user_id=None (anonymous)
        result = await playground_service.execute_playground(
            sample_playground.id, request, None
        )

        assert result.is_success is True


class TestExecutePlaygroundRoute:
    """Tests for execute_playground route."""

    @pytest.mark.asyncio
    async def test_execute_playground_route_success(
        self, playground_service, mock_playground_repo, mock_history_repo, sample_playground
    ):
        """Test execute_playground route success."""
        from code_tutor.playground.interface.routes import execute_playground
        from code_tutor.playground.application.dto import (
            ExecutePlaygroundRequest,
            ExecutionResponse,
        )
        from code_tutor.identity.application.dto import UserResponse

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)
        mock_history_repo.save = AsyncMock()

        request = ExecutePlaygroundRequest(code="print('hello')")

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = sample_playground.owner_id

        result = await execute_playground(
            playground_id=sample_playground.id,
            request=request,
            current_user=mock_user,
            service=playground_service,
        )

        assert isinstance(result, ExecutionResponse)
        assert result.is_success is True

    @pytest.mark.asyncio
    async def test_execute_playground_route_not_found(
        self, playground_service, mock_playground_repo
    ):
        """Test execute_playground route when playground not found."""
        from code_tutor.playground.interface.routes import execute_playground
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=None)

        request = ExecutePlaygroundRequest(code="print('hello')")

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            await execute_playground(
                playground_id=uuid4(),
                request=request,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_execute_playground_route_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test execute_playground route when forbidden."""
        from code_tutor.playground.interface.routes import execute_playground
        from code_tutor.playground.application.dto import ExecutePlaygroundRequest
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        owner_id = uuid4()
        private_playground = Playground.create(
            owner_id=owner_id,
            title="Private",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        mock_playground_repo.get_by_id = AsyncMock(return_value=private_playground)

        request = ExecutePlaygroundRequest(code="print('hello')")

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await execute_playground(
                playground_id=private_playground.id,
                request=request,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_execute_playground_route_anonymous(
        self, playground_service, mock_playground_repo, mock_history_repo, sample_playground
    ):
        """Test execute_playground route with anonymous user."""
        from code_tutor.playground.interface.routes import execute_playground
        from code_tutor.playground.application.dto import (
            ExecutePlaygroundRequest,
            ExecutionResponse,
        )

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        async def save_playground(playground):
            return playground

        mock_playground_repo.save = AsyncMock(side_effect=save_playground)
        mock_history_repo.save = AsyncMock()

        request = ExecutePlaygroundRequest(code="print('hello')")

        result = await execute_playground(
            playground_id=sample_playground.id,
            request=request,
            current_user=None,  # Anonymous
            service=playground_service,
        )

        assert isinstance(result, ExecutionResponse)
        assert result.is_success is True


class TestForkPlaygroundRouteAdditional:
    """Additional tests for fork_playground route."""

    @pytest.mark.asyncio
    async def test_fork_playground_route_forbidden(
        self, playground_service, mock_playground_repo
    ):
        """Test fork_playground route when forbidden."""
        from code_tutor.playground.interface.routes import fork_playground
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        owner_id = uuid4()
        private_playground = Playground.create(
            owner_id=owner_id,
            title="Private",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        mock_playground_repo.get_by_id = AsyncMock(return_value=private_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await fork_playground(
                playground_id=private_playground.id,
                request=None,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403


class TestRegenerateShareCodeRouteAdditional:
    """Additional tests for regenerate_share_code route."""

    @pytest.mark.asyncio
    async def test_regenerate_share_code_route_forbidden(
        self, playground_service, mock_playground_repo, sample_playground
    ):
        """Test regenerate_share_code route when forbidden."""
        from code_tutor.playground.interface.routes import regenerate_share_code
        from code_tutor.identity.application.dto import UserResponse
        from fastapi import HTTPException

        mock_playground_repo.get_by_id = AsyncMock(return_value=sample_playground)

        mock_user = MagicMock(spec=UserResponse)
        mock_user.id = uuid4()  # Different user

        with pytest.raises(HTTPException) as exc_info:
            await regenerate_share_code(
                playground_id=sample_playground.id,
                current_user=mock_user,
                service=playground_service,
            )

        assert exc_info.value.status_code == 403
