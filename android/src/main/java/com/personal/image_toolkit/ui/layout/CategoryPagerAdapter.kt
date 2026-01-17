package com.personal.nglab.ui

import androidx.fragment.app.Fragment
import androidx.viewpager2.adapter.FragmentStateAdapter
import com.personal.nglab.ui.tabs.ConvertFragment
import com.personal.nglab.ui.tabs.DeleteFragment
import com.personal.nglab.ui.tabs.DriveSyncFragment
import com.personal.nglab.ui.tabs.GenerateFragment
import com.personal.nglab.ui.tabs.ImageCrawlerFragment
import com.personal.nglab.ui.tabs.ImageExtractorFragment
import com.personal.nglab.ui.tabs.MergeFragment
import com.personal.nglab.ui.tabs.ReverseImageSearchFragment
import com.personal.nglab.ui.tabs.TrainFragment
import com.personal.nglab.ui.tabs.WallpaperFragment
import com.personal.nglab.ui.tabs.WebRequestsFragment

enum class AppCategory {
    SYSTEM_TOOLS,
    WEB_INTEGRATION,
    DEEP_LEARNING,
    DATABASE, // Placeholder
}

class CategoryPagerAdapter(
    fragment: Fragment,
    private val category: AppCategory,
) : FragmentStateAdapter(fragment) {
    override fun getItemCount(): Int =
        when (category) {
            AppCategory.SYSTEM_TOOLS -> 5
            AppCategory.WEB_INTEGRATION -> 4
            AppCategory.DEEP_LEARNING -> 2
            AppCategory.DATABASE -> 0
        }

    override fun createFragment(position: Int): Fragment {
        return when (category) {
            AppCategory.SYSTEM_TOOLS ->
                when (position) {
                    0 -> ConvertFragment()
                    1 -> MergeFragment()
                    2 -> DeleteFragment()
                    3 -> ImageExtractorFragment()
                    else -> WallpaperFragment()
                }
            AppCategory.WEB_INTEGRATION ->
                when (position) {
                    0 -> ImageCrawlerFragment()
                    1 -> WebRequestsFragment()
                    2 -> DriveSyncFragment()
                    else -> ReverseImageSearchFragment()
                }
            AppCategory.DEEP_LEARNING ->
                when (position) {
                    0 -> TrainFragment()
                    else -> GenerateFragment()
                }
            else -> Fragment() // Placeholder
        }
    }

    fun getTabTitle(position: Int): String {
        return when (category) {
            AppCategory.SYSTEM_TOOLS ->
                when (position) {
                    0 -> "Convert"
                    1 -> "Merge"
                    2 -> "Delete"
                    3 -> "Extractor"
                    else -> "Wallpaper"
                }
            AppCategory.WEB_INTEGRATION ->
                when (position) {
                    0 -> "Crawler"
                    1 -> "Web Req"
                    2 -> "Cloud Sync"
                    else -> "Rev Search"
                }
            AppCategory.DEEP_LEARNING ->
                when (position) {
                    0 -> "Train"
                    else -> "Generate"
                }
            else -> "Tab $position"
        }
    }
}
