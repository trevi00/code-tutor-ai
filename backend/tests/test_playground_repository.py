"""Tests for Playground Repository implementations."""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from code_tutor.playground.domain.entities import (
    CodeTemplate,
    ExecutionHistory,
    Playground,
)
from code_tutor.playground.domain.value_objects import (
    PlaygroundLanguage,
    PlaygroundVisibility,
    TemplateCategory,
)
from code_tutor.playground.infrastructure.models import (
    CodeTemplateModel,
    ExecutionHistoryModel,
    PlaygroundModel,
)
from code_tutor.playground.infrastructure.repository import (
    SQLAlchemyExecutionHistoryRepository,
    SQLAlchemyPlaygroundRepository,
    SQLAlchemyTemplateRepository,
)
from code_tutor.shared.infrastructure.database import Base


@pytest_asyncio.fixture
async def async_engine():
    """Create async engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    # Enable foreign keys for SQLite
    @event.listens_for(engine.sync_engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=OFF")  # Disable for testing
        cursor.close()

    async with engine.begin() as conn:
        # Create users table for foreign key reference
        await conn.execute(
            text("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                email TEXT,
                hashed_password TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT
            )
        """)
        )

        # Create playgrounds table
        await conn.execute(
            text("""
            CREATE TABLE playgrounds (
                id TEXT PRIMARY KEY,
                owner_id TEXT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                code TEXT NOT NULL,
                language TEXT DEFAULT 'python',
                visibility TEXT DEFAULT 'private',
                share_code TEXT UNIQUE NOT NULL,
                stdin TEXT DEFAULT '',
                is_forked INTEGER DEFAULT 0,
                forked_from_id TEXT,
                run_count INTEGER DEFAULT 0,
                fork_count INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
        """)
        )

        # Create code_templates table
        await conn.execute(
            text("""
            CREATE TABLE code_templates (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                code TEXT NOT NULL,
                language TEXT DEFAULT 'python',
                category TEXT DEFAULT 'snippet',
                tags TEXT DEFAULT '',
                usage_count INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        )

        # Create playground_executions table
        await conn.execute(
            text("""
            CREATE TABLE playground_executions (
                id TEXT PRIMARY KEY,
                playground_id TEXT NOT NULL,
                user_id TEXT,
                code TEXT NOT NULL,
                stdin TEXT DEFAULT '',
                stdout TEXT DEFAULT '',
                stderr TEXT DEFAULT '',
                exit_code INTEGER DEFAULT 0,
                execution_time_ms REAL DEFAULT 0.0,
                is_success INTEGER DEFAULT 0,
                executed_at TEXT
            )
        """)
        )

    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def async_session(async_engine):
    """Create async session for testing."""
    async_session_maker = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def sample_owner_id():
    """Generate a sample owner ID."""
    return uuid4()


class TestSQLAlchemyPlaygroundRepository:
    """Tests for SQLAlchemyPlaygroundRepository."""

    @pytest_asyncio.fixture
    async def repository(self, async_session):
        """Create repository with session."""
        return SQLAlchemyPlaygroundRepository(async_session)

    @pytest.fixture
    def sample_playground(self, sample_owner_id):
        """Create a sample playground entity."""
        return Playground.create(
            owner_id=sample_owner_id,
            title="Test Playground",
            code="print('Hello, World!')",
            language=PlaygroundLanguage.PYTHON,
            description="A test playground",
            visibility=PlaygroundVisibility.PRIVATE,
            stdin="test input",
        )

    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, repository, sample_playground):
        """Test saving and retrieving a playground."""
        saved = await repository.save(sample_playground)

        assert saved.id == sample_playground.id
        assert saved.title == "Test Playground"
        assert saved.code == "print('Hello, World!')"
        assert saved.language == PlaygroundLanguage.PYTHON

        retrieved = await repository.get_by_id(sample_playground.id)

        assert retrieved is not None
        assert retrieved.id == sample_playground.id
        assert retrieved.title == "Test Playground"
        assert retrieved.language == PlaygroundLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository):
        """Test getting non-existent playground."""
        result = await repository.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_by_share_code(self, repository, sample_playground):
        """Test retrieving playground by share code."""
        await repository.save(sample_playground)

        retrieved = await repository.get_by_share_code(sample_playground.share_code)

        assert retrieved is not None
        assert retrieved.id == sample_playground.id
        assert retrieved.share_code == sample_playground.share_code

    @pytest.mark.asyncio
    async def test_get_by_share_code_not_found(self, repository):
        """Test getting playground by non-existent share code."""
        result = await repository.get_by_share_code("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_playground(self, repository, sample_playground):
        """Test updating an existing playground."""
        await repository.save(sample_playground)

        sample_playground.update(
            title="Updated Title",
            description="Updated description",
            code="print('Updated!')",
        )

        updated = await repository.save(sample_playground)

        assert updated.title == "Updated Title"
        assert updated.description == "Updated description"
        assert updated.code == "print('Updated!')"

        retrieved = await repository.get_by_id(sample_playground.id)
        assert retrieved.title == "Updated Title"

    @pytest.mark.asyncio
    async def test_delete_playground(self, repository, sample_playground):
        """Test deleting a playground."""
        await repository.save(sample_playground)

        retrieved = await repository.get_by_id(sample_playground.id)
        assert retrieved is not None

        await repository.delete(sample_playground.id)

        retrieved = await repository.get_by_id(sample_playground.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, repository):
        """Test deleting non-existent playground (should not raise)."""
        await repository.delete(uuid4())

    @pytest.mark.asyncio
    async def test_get_user_playgrounds(self, repository, sample_owner_id):
        """Test getting playgrounds by owner."""
        playgrounds = []
        for i in range(3):
            pg = Playground.create(
                owner_id=sample_owner_id,
                title=f"Playground {i}",
                code=f"print({i})",
                language=PlaygroundLanguage.PYTHON,
            )
            await repository.save(pg)
            playgrounds.append(pg)

        result = await repository.get_user_playgrounds(sample_owner_id)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_user_playgrounds_pagination(self, repository, sample_owner_id):
        """Test pagination for user playgrounds."""
        for i in range(5):
            pg = Playground.create(
                owner_id=sample_owner_id,
                title=f"Playground {i}",
                code=f"print({i})",
                language=PlaygroundLanguage.PYTHON,
            )
            await repository.save(pg)

        result = await repository.get_user_playgrounds(
            sample_owner_id, limit=2, offset=0
        )
        assert len(result) == 2

        result = await repository.get_user_playgrounds(
            sample_owner_id, limit=2, offset=2
        )
        assert len(result) == 2

        result = await repository.get_user_playgrounds(
            sample_owner_id, limit=2, offset=4
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_get_user_playgrounds_empty(self, repository):
        """Test getting playgrounds for user with none."""
        result = await repository.get_user_playgrounds(uuid4())
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_public_playgrounds(self, repository, sample_owner_id):
        """Test getting public playgrounds."""
        # Create public playground
        public_pg = Playground.create(
            owner_id=sample_owner_id,
            title="Public Playground",
            code="print('public')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(public_pg)

        # Create private playground
        private_pg = Playground.create(
            owner_id=sample_owner_id,
            title="Private Playground",
            code="print('private')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        await repository.save(private_pg)

        result = await repository.get_public_playgrounds()

        assert len(result) == 1
        assert result[0].visibility == PlaygroundVisibility.PUBLIC

    @pytest.mark.asyncio
    async def test_get_public_playgrounds_with_language_filter(
        self, repository, sample_owner_id
    ):
        """Test filtering public playgrounds by language."""
        # Python public
        python_pg = Playground.create(
            owner_id=sample_owner_id,
            title="Python Public",
            code="print('hello')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(python_pg)

        # JavaScript public
        js_pg = Playground.create(
            owner_id=sample_owner_id,
            title="JS Public",
            code="console.log('hello')",
            language=PlaygroundLanguage.JAVASCRIPT,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(js_pg)

        result = await repository.get_public_playgrounds(
            language=PlaygroundLanguage.PYTHON
        )

        assert len(result) == 1
        assert result[0].language == PlaygroundLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_get_popular_playgrounds(self, repository, sample_owner_id):
        """Test getting popular playgrounds by run count."""
        # Create playgrounds with different run counts
        for run_count in [10, 50, 5]:
            pg = Playground.create(
                owner_id=sample_owner_id,
                title=f"Playground with {run_count} runs",
                code="print('test')",
                language=PlaygroundLanguage.PYTHON,
                visibility=PlaygroundVisibility.PUBLIC,
            )
            # Manually set run_count
            for _ in range(run_count):
                pg.increment_run_count()
            await repository.save(pg)

        result = await repository.get_popular_playgrounds(limit=3)

        assert len(result) == 3
        # Should be ordered by run_count descending
        assert result[0].run_count == 50
        assert result[1].run_count == 10
        assert result[2].run_count == 5

    @pytest.mark.asyncio
    async def test_search_playgrounds_by_title(self, repository, sample_owner_id):
        """Test searching playgrounds by title."""
        pg1 = Playground.create(
            owner_id=sample_owner_id,
            title="Python Algorithm",
            code="print('algorithm')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(pg1)

        pg2 = Playground.create(
            owner_id=sample_owner_id,
            title="JavaScript Demo",
            code="console.log('demo')",
            language=PlaygroundLanguage.JAVASCRIPT,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(pg2)

        result = await repository.search_playgrounds("Algorithm")

        assert len(result) == 1
        assert "Algorithm" in result[0].title

    @pytest.mark.asyncio
    async def test_search_playgrounds_by_description(self, repository, sample_owner_id):
        """Test searching playgrounds by description."""
        pg = Playground.create(
            owner_id=sample_owner_id,
            title="Test",
            code="print('test')",
            language=PlaygroundLanguage.PYTHON,
            description="Binary search implementation",
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(pg)

        result = await repository.search_playgrounds("binary")

        assert len(result) == 1
        assert "binary" in result[0].description.lower()

    @pytest.mark.asyncio
    async def test_search_playgrounds_with_language_filter(
        self, repository, sample_owner_id
    ):
        """Test searching playgrounds with language filter."""
        pg1 = Playground.create(
            owner_id=sample_owner_id,
            title="Algorithm Python",
            code="print('test')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(pg1)

        pg2 = Playground.create(
            owner_id=sample_owner_id,
            title="Algorithm JavaScript",
            code="console.log('test')",
            language=PlaygroundLanguage.JAVASCRIPT,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(pg2)

        result = await repository.search_playgrounds(
            "Algorithm", language=PlaygroundLanguage.PYTHON
        )

        assert len(result) == 1
        assert result[0].language == PlaygroundLanguage.PYTHON

    @pytest.mark.asyncio
    async def test_search_excludes_private(self, repository, sample_owner_id):
        """Test that search only returns public playgrounds."""
        public_pg = Playground.create(
            owner_id=sample_owner_id,
            title="Public Algorithm",
            code="print('public')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PUBLIC,
        )
        await repository.save(public_pg)

        private_pg = Playground.create(
            owner_id=sample_owner_id,
            title="Private Algorithm",
            code="print('private')",
            language=PlaygroundLanguage.PYTHON,
            visibility=PlaygroundVisibility.PRIVATE,
        )
        await repository.save(private_pg)

        result = await repository.search_playgrounds("Algorithm")

        assert len(result) == 1
        assert result[0].visibility == PlaygroundVisibility.PUBLIC


class TestSQLAlchemyTemplateRepository:
    """Tests for SQLAlchemyTemplateRepository."""

    @pytest_asyncio.fixture
    async def repository(self, async_session):
        """Create repository with session."""
        return SQLAlchemyTemplateRepository(async_session)

    @pytest.fixture
    def sample_template(self):
        """Create a sample template entity."""
        return CodeTemplate(
            id=uuid4(),
            title="Two Pointers",
            description="Two pointers pattern template",
            code="def two_pointers(arr):\n    left, right = 0, len(arr) - 1",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.ALGORITHM,
            tags=["algorithm", "array"],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_save_and_get_by_id(self, repository, sample_template):
        """Test saving and retrieving a template."""
        saved = await repository.save(sample_template)

        assert saved.id == sample_template.id
        assert saved.title == "Two Pointers"
        assert saved.language == PlaygroundLanguage.PYTHON
        assert saved.category == TemplateCategory.ALGORITHM

        retrieved = await repository.get_by_id(sample_template.id)

        assert retrieved is not None
        assert retrieved.id == sample_template.id
        assert retrieved.title == "Two Pointers"
        assert retrieved.tags == ["algorithm", "array"]

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, repository):
        """Test getting non-existent template."""
        result = await repository.get_by_id(uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self, repository):
        """Test getting all templates."""
        templates = [
            CodeTemplate(
                id=uuid4(),
                title=f"Template {i}",
                description=f"Description {i}",
                code=f"code {i}",
                language=PlaygroundLanguage.PYTHON,
                category=TemplateCategory.SNIPPET,
                tags=[],
                usage_count=0,
                created_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        for t in templates:
            await repository.save(t)

        result = await repository.get_all()

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_all_with_category_filter(self, repository):
        """Test filtering templates by category."""
        algo_template = CodeTemplate(
            id=uuid4(),
            title="Algorithm Template",
            description="Algorithm",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.ALGORITHM,
            tags=[],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )
        await repository.save(algo_template)

        snippet_template = CodeTemplate(
            id=uuid4(),
            title="Snippet Template",
            description="Snippet",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.SNIPPET,
            tags=[],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )
        await repository.save(snippet_template)

        result = await repository.get_all(category=TemplateCategory.ALGORITHM)

        assert len(result) == 1
        assert result[0].category == TemplateCategory.ALGORITHM

    @pytest.mark.asyncio
    async def test_get_all_with_language_filter(self, repository):
        """Test filtering templates by language."""
        python_template = CodeTemplate(
            id=uuid4(),
            title="Python Template",
            description="Python",
            code="print('hello')",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.SNIPPET,
            tags=[],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )
        await repository.save(python_template)

        js_template = CodeTemplate(
            id=uuid4(),
            title="JS Template",
            description="JavaScript",
            code="console.log('hello')",
            language=PlaygroundLanguage.JAVASCRIPT,
            category=TemplateCategory.SNIPPET,
            tags=[],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )
        await repository.save(js_template)

        result = await repository.get_all(language=PlaygroundLanguage.JAVASCRIPT)

        assert len(result) == 1
        assert result[0].language == PlaygroundLanguage.JAVASCRIPT

    @pytest.mark.asyncio
    async def test_get_popular(self, repository):
        """Test getting popular templates by usage count."""
        for usage_count in [100, 50, 200]:
            template = CodeTemplate(
                id=uuid4(),
                title=f"Template {usage_count}",
                description="Description",
                code="code",
                language=PlaygroundLanguage.PYTHON,
                category=TemplateCategory.SNIPPET,
                tags=[],
                usage_count=usage_count,
                created_at=datetime.now(timezone.utc),
            )
            await repository.save(template)

        result = await repository.get_popular(limit=3)

        assert len(result) == 3
        # Should be ordered by usage_count descending
        assert result[0].usage_count == 200
        assert result[1].usage_count == 100
        assert result[2].usage_count == 50

    @pytest.mark.asyncio
    async def test_tags_serialization(self, repository):
        """Test that tags are properly serialized/deserialized."""
        template = CodeTemplate(
            id=uuid4(),
            title="Tagged Template",
            description="Template with tags",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.ALGORITHM,
            tags=["tag1", "tag2", "tag3"],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )

        await repository.save(template)
        retrieved = await repository.get_by_id(template.id)

        assert retrieved is not None
        assert retrieved.tags == ["tag1", "tag2", "tag3"]

    @pytest.mark.asyncio
    async def test_empty_tags(self, repository):
        """Test template with empty tags."""
        template = CodeTemplate(
            id=uuid4(),
            title="No Tags",
            description="Template without tags",
            code="code",
            language=PlaygroundLanguage.PYTHON,
            category=TemplateCategory.SNIPPET,
            tags=[],
            usage_count=0,
            created_at=datetime.now(timezone.utc),
        )

        await repository.save(template)
        retrieved = await repository.get_by_id(template.id)

        assert retrieved is not None
        assert retrieved.tags == []


class TestSQLAlchemyExecutionHistoryRepository:
    """Tests for SQLAlchemyExecutionHistoryRepository."""

    @pytest_asyncio.fixture
    async def repository(self, async_session):
        """Create repository with session."""
        return SQLAlchemyExecutionHistoryRepository(async_session)

    @pytest_asyncio.fixture
    async def playground_repo(self, async_session):
        """Create playground repository for setup."""
        return SQLAlchemyPlaygroundRepository(async_session)

    @pytest.fixture
    def sample_playground(self, sample_owner_id):
        """Create sample playground for execution history."""
        return Playground.create(
            owner_id=sample_owner_id,
            title="Test Playground",
            code="print('test')",
            language=PlaygroundLanguage.PYTHON,
        )

    @pytest.fixture
    def sample_history(self, sample_playground, sample_owner_id):
        """Create sample execution history."""
        return ExecutionHistory(
            id=uuid4(),
            playground_id=sample_playground.id,
            user_id=sample_owner_id,
            code="print('test')",
            stdin="",
            stdout="test\n",
            stderr="",
            exit_code=0,
            execution_time_ms=50.0,
            is_success=True,
            executed_at=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_save_execution_history(
        self, repository, playground_repo, sample_playground, sample_history
    ):
        """Test saving execution history."""
        await playground_repo.save(sample_playground)
        saved = await repository.save(sample_history)

        assert saved.id == sample_history.id
        assert saved.playground_id == sample_playground.id
        assert saved.stdout == "test\n"
        assert saved.is_success is True

    @pytest.mark.asyncio
    async def test_get_playground_history(
        self, repository, playground_repo, sample_playground, sample_owner_id
    ):
        """Test getting execution history for a playground."""
        await playground_repo.save(sample_playground)

        # Create multiple history entries
        for i in range(3):
            history = ExecutionHistory(
                id=uuid4(),
                playground_id=sample_playground.id,
                user_id=sample_owner_id,
                code=f"print({i})",
                stdin="",
                stdout=f"{i}\n",
                stderr="",
                exit_code=0,
                execution_time_ms=10.0 + i,
                is_success=True,
                executed_at=datetime.now(timezone.utc),
            )
            await repository.save(history)

        result = await repository.get_playground_history(sample_playground.id)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_playground_history_limit(
        self, repository, playground_repo, sample_playground, sample_owner_id
    ):
        """Test pagination limit for playground history."""
        await playground_repo.save(sample_playground)

        for i in range(5):
            history = ExecutionHistory(
                id=uuid4(),
                playground_id=sample_playground.id,
                user_id=sample_owner_id,
                code=f"print({i})",
                stdin="",
                stdout=f"{i}\n",
                stderr="",
                exit_code=0,
                execution_time_ms=10.0,
                is_success=True,
                executed_at=datetime.now(timezone.utc),
            )
            await repository.save(history)

        result = await repository.get_playground_history(sample_playground.id, limit=2)

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_get_playground_history_empty(self, repository):
        """Test getting history for playground with none."""
        result = await repository.get_playground_history(uuid4())
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_get_user_history(
        self, repository, playground_repo, sample_owner_id
    ):
        """Test getting execution history for a user."""
        # Create a playground first
        playground = Playground.create(
            owner_id=sample_owner_id,
            title="User Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )
        await playground_repo.save(playground)

        # Create history entries
        for i in range(3):
            history = ExecutionHistory(
                id=uuid4(),
                playground_id=playground.id,
                user_id=sample_owner_id,
                code=f"print({i})",
                stdin="",
                stdout=f"{i}\n",
                stderr="",
                exit_code=0,
                execution_time_ms=10.0,
                is_success=True,
                executed_at=datetime.now(timezone.utc),
            )
            await repository.save(history)

        result = await repository.get_user_history(sample_owner_id)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_user_history_limit(
        self, repository, playground_repo, sample_owner_id
    ):
        """Test pagination limit for user history."""
        playground = Playground.create(
            owner_id=sample_owner_id,
            title="Limit Test",
            code="code",
            language=PlaygroundLanguage.PYTHON,
        )
        await playground_repo.save(playground)

        for i in range(5):
            history = ExecutionHistory(
                id=uuid4(),
                playground_id=playground.id,
                user_id=sample_owner_id,
                code=f"print({i})",
                stdin="",
                stdout=f"{i}\n",
                stderr="",
                exit_code=0,
                execution_time_ms=10.0,
                is_success=True,
                executed_at=datetime.now(timezone.utc),
            )
            await repository.save(history)

        result = await repository.get_user_history(sample_owner_id, limit=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_get_user_history_empty(self, repository):
        """Test getting history for user with none."""
        result = await repository.get_user_history(uuid4())
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_execution_with_error(
        self, repository, playground_repo, sample_playground, sample_owner_id
    ):
        """Test saving execution history with error."""
        await playground_repo.save(sample_playground)

        history = ExecutionHistory(
            id=uuid4(),
            playground_id=sample_playground.id,
            user_id=sample_owner_id,
            code="print(undefined_var)",
            stdin="",
            stdout="",
            stderr="NameError: name 'undefined_var' is not defined",
            exit_code=1,
            execution_time_ms=25.0,
            is_success=False,
            executed_at=datetime.now(timezone.utc),
        )

        saved = await repository.save(history)

        assert saved.is_success is False
        assert saved.exit_code == 1
        assert "NameError" in saved.stderr

    @pytest.mark.asyncio
    async def test_execution_without_user(
        self, repository, playground_repo, sample_playground
    ):
        """Test saving execution history without user (anonymous)."""
        await playground_repo.save(sample_playground)

        history = ExecutionHistory(
            id=uuid4(),
            playground_id=sample_playground.id,
            user_id=None,
            code="print('anonymous')",
            stdin="",
            stdout="anonymous\n",
            stderr="",
            exit_code=0,
            execution_time_ms=15.0,
            is_success=True,
            executed_at=datetime.now(timezone.utc),
        )

        saved = await repository.save(history)

        assert saved.user_id is None
        assert saved.is_success is True
