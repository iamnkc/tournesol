import React, { useState, useEffect, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { useLocation, useHistory, Link } from 'react-router-dom';
import { Box, Button, Divider } from '@mui/material';

import type {
  ContributorRating,
  PaginatedContributorRatingList,
} from 'src/services/openapi';
import Pagination from 'src/components/Pagination';
import VideoList from 'src/features/videos/VideoList';
import { UsersService } from 'src/services/openapi';
import { ContentBox, ContentHeader, LoaderWrapper } from 'src/components';
import {
  PublicStatusAction,
  RatingsContext,
} from 'src/features/videos/PublicStatusAction';
import RatingsFilter from 'src/features/ratings/RatingsFilter';
import { scrollToTop } from 'src/utils/ui';

const NoRatingMessage = ({ hasFilter }: { hasFilter: boolean }) => {
  const { t } = useTranslation();
  return (
    <>
      <Divider />
      {hasFilter ? (
        <Box my={2}>{t('ratings.noVideoCorrespondsToFilter')}</Box>
      ) : (
        <>
          <Box my={2}>
            {t('ratings.noVideoComparedYet')}
            {' 🥺'}
          </Box>
          <Button
            component={Link}
            to="/comparison"
            variant="contained"
            color="primary"
          >
            {t('ratings.compareVideosButton')}
          </Button>
        </>
      )}
    </>
  );
};

const VideoRatingsPage = () => {
  const [ratings, setRatings] = useState<PaginatedContributorRatingList>({});
  const [isLoading, setIsLoading] = useState(true);
  const location = useLocation();
  const history = useHistory();
  const { t } = useTranslation();
  const searchParams = new URLSearchParams(location.search);
  const limit = 20;
  const offset = Number(searchParams.get('offset') || 0);
  const videoCount = ratings.count || 0;
  const hasFilter = searchParams.get('isPublic') != null;

  const handleOffsetChange = (newOffset: number) => {
    searchParams.set('offset', newOffset.toString());
    history.push({ search: searchParams.toString() });
    scrollToTop();
  };

  const loadData = useCallback(async () => {
    setIsLoading(true);
    const urlParams = new URLSearchParams(location.search);
    const isPublicParam = urlParams.get('isPublic');
    const isPublic = isPublicParam ? isPublicParam === 'true' : undefined;
    const response = await UsersService.usersMeContributorRatingsList({
      limit,
      offset,
      isPublic,
    });
    setRatings(response);
    setIsLoading(false);
  }, [offset, location.search]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const videos = (ratings.results || []).map(
    (rating: ContributorRating) => rating.video
  );
  const idToRating = Object.fromEntries(
    (ratings.results || []).map((rating) => [rating.video.video_id, rating])
  );
  const getRating = (videoId: string) => idToRating[videoId];

  const onRatingChange = (newRating: ContributorRating | undefined) => {
    if (newRating) {
      setRatings((prevRatings) => {
        const updatedResults = (prevRatings.results || []).map((rating) =>
          rating.video.video_id === newRating.video.video_id
            ? newRating
            : rating
        );
        return { ...prevRatings, results: updatedResults };
      });
    } else {
      // All ratings have been updated.
      if (hasFilter) {
        // A filter had been selected. Let's reset the filter to reload the list.
        searchParams.delete('isPublic');
        history.push({ search: searchParams.toString() });
      } else {
        // No filter is selected. Let's simply refresh the list.
        loadData();
      }
    }
  };

  return (
    <RatingsContext.Provider
      value={{
        getContributorRating: getRating,
        onChange: onRatingChange,
      }}
    >
      <ContentHeader title={t('myRatedVideosPage.title')} />
      <ContentBox noMinPaddingX maxWidth="md">
        <Box px={{ xs: 2, sm: 0 }}>
          <RatingsFilter />
        </Box>
        <LoaderWrapper isLoading={isLoading}>
          <VideoList
            videos={videos}
            settings={[PublicStatusAction]}
            emptyMessage={<NoRatingMessage hasFilter={hasFilter} />}
          />
        </LoaderWrapper>
        {!isLoading && videoCount > 0 && (
          <Pagination
            offset={offset}
            count={videoCount}
            onOffsetChange={handleOffsetChange}
            limit={limit}
          />
        )}
      </ContentBox>
    </RatingsContext.Provider>
  );
};

export default VideoRatingsPage;
